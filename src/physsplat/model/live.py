"""LiveSim: the learned simulator as a steppable object.

One instance holds the body state and velocity history; step() advances one
DT with an optional external force. Used three ways: the eval rollout loop,
the Phase 5 WebSocket demo server (closed-loop grabs), and as the reference
semantics for the Phase 7 TypeScript port.
"""

import numpy as np
import torch

from ..common import constants as C
from ..common.actions import action_feature
from .graph import build_edges, edge_features
from .integrator import external_accels, quat_to_matrix, step

EDGE_BUCKET = 4096
NODE_BUCKET = 256   # pad node count so per-scene shapes don't multiply


class LiveSim:
    # Process-wide defaults for the analytic rules (scripts/evaluate.py
    # --free-flight / --energy flip them so the scorecard can be run both
    # ways). Both are physics, not cleanup: they say what the residual is
    # not allowed to claim, and the demo runs with them on.
    DEFAULT_FREE_FLIGHT = False
    DEFAULT_ENERGY_RULE = False

    def __init__(self, model, normalizer, scene: dict, init: dict, device="cpu",
                 ground_guard: bool = False, free_flight: bool | None = None,
                 energy_rule: bool | None = None):
        """scene: offsets_list, mass, inertia (numpy).
        init: pos/quat/linvel/angvel, each (HISTORY, B, ...) numpy warmup.
        ground_guard: analytic non-penetration cleanup for demos (lift a body
        whose particles dip below z=0, cancel downward velocity). Off for
        evaluation so the scorecard measures the model, not the guard.
        free_flight: zero the learned residual for bodies touching nothing
        (see step); an analytic rule, not a guard, so it may be evaluated.
        energy_rule: scale the residual down when it would add mechanical
        energy beyond the applied force's work (see step)."""
        self.model, self.norm, self.device = model, normalizer, device
        self.ground_guard = ground_guard
        self.free_flight = (LiveSim.DEFAULT_FREE_FLIGHT if free_flight is None
                            else free_flight)
        self.energy_rule = (LiveSim.DEFAULT_ENERGY_RULE if energy_rule is None
                            else energy_rule)
        B = self.B = len(scene["mass"])
        self.offsets = [torch.tensor(o, dtype=torch.float32, device=device)
                        for o in scene["offsets_list"]]
        counts = np.array([len(o) for o in self.offsets])
        self.body_ids_np = np.repeat(np.arange(B), counts)
        # Dummy nodes + dummy body (MPS shape bucketing, see train.loop):
        # pad the node count up to a bucket multiple (at least one dummy)
        # so scenes with different particle counts share kernel shapes.
        N = len(self.body_ids_np)
        self.n_pad = max(1, -(-(N + 1) // NODE_BUCKET) * NODE_BUCKET - N)
        self.body_ids = torch.tensor(
            np.concatenate([self.body_ids_np, np.full(self.n_pad, B)]),
            device=device)
        self.mass = torch.cat(
            [torch.tensor(scene["mass"], dtype=torch.float32),
             torch.tensor([0.01])]).to(device)
        self.inertia = torch.cat(
            [torch.tensor(scene["inertia"], dtype=torch.float32),
             torch.full((1, 3), 1e-6)]).to(device)
        self.body_scalars = normalizer.body_scalars(self.mass, self.inertia)

        self.pos = torch.tensor(init["pos"][-1], dtype=torch.float32, device=device)
        self.quat = torch.tensor(init["quat"][-1], dtype=torch.float32, device=device)
        self.lin_hist = [torch.tensor(v, dtype=torch.float32, device=device)
                         for v in init["linvel"]][-C.HISTORY:]
        self.ang_hist = [torch.tensor(v, dtype=torch.float32, device=device)
                         for v in init["angvel"]][-C.HISTORY:]
        self.quat_hist = [torch.tensor(q, dtype=torch.float32, device=device)
                          for q in init["quat"]][-C.HISTORY:]

    def _energy(self, pos, quat, linvel, angvel) -> torch.Tensor:
        """Mechanical energy of the real bodies (J): kinetic + gravitational."""
        m = self.mass[:self.B, None]
        I = self.inertia[:self.B]
        R = quat_to_matrix(quat)
        w_body = torch.einsum("bij,bi->bj", R, angvel)      # R^T w
        ke = 0.5 * (m * linvel.pow(2)).sum() + 0.5 * (I * w_body.pow(2)).sum()
        return ke + (m[:, 0] * C.GRAVITY * pos[:, 2]).sum()

    def _limit_energy(self, residual, act_body, act_point, act_force):
        """No free energy: contact with static things cannot add mechanical
        energy to a scene, and an applied force can add only the work it
        does. The network's residual on out-of-distribution (reconstructed)
        piles violated that, standing pencils up on their own. Trial-step
        the scene; if the energy rises by more than the action's work plus a
        small allowance for pushing out of overlaps, scale the residual down
        until it does not (bisection, 5 rounds)."""
        lin, ang = self.lin_hist[-1], self.ang_hist[-1]
        E0 = self._energy(self.pos, self.quat, lin, ang)
        work = torch.zeros((), device=self.device)
        if act_body >= 0:
            r = act_point - self.pos[act_body]
            v_point = lin[act_body] + torch.linalg.cross(ang[act_body], r)
            work = torch.dot(act_force, v_point) * C.DT
        allow = work.clamp_min(0) + 5e-7

        def gain(scale: float) -> torch.Tensor:
            ext_l, ext_a = external_accels(
                self.pos, self.quat, self.mass, self.inertia,
                act_body, act_point, act_force)
            p, q, lv, av = step(self.pos, self.quat, lin, ang,
                                residual * scale, ext_l, ext_a)
            return self._energy(p, q, lv, av) - E0

        if float(gain(1.0)) <= float(allow):
            return residual
        lo, hi = (0.0, 0.0) if float(gain(0.0)) > float(allow) else (0.0, 1.0)
        for _ in range(5):
            if hi <= lo:
                break
            mid = 0.5 * (lo + hi)
            if float(gain(mid)) > float(allow):
                hi = mid
            else:
                lo = mid
        self.energy_scale = lo
        return residual * lo

    def particles_world(self) -> torch.Tensor:
        R = quat_to_matrix(self.quat)
        return torch.cat(
            [self.offsets[b] @ R[b].T + self.pos[b] for b in range(self.B)])

    def step(self, act_body: int = -1, act_point=None, act_force=None):
        """Advance one DT. act_point/act_force: world-space numpy (3,).

        Differentiable when autograd is enabled (rollout fine-tuning
        backpropagates through several steps); inference callers wrap in
        torch.no_grad(). Graph indices and action weights are computed on
        detached copies (they are discrete / non-differentiable anyway)."""
        parts = self.particles_world()
        vh = []
        for h in range(C.HISTORY):
            Rh = quat_to_matrix(self.quat_hist[h])
            v = torch.cat([
                self.lin_hist[h][b] + torch.linalg.cross(
                    self.ang_hist[h][b].expand(len(self.offsets[b]), 3),
                    self.offsets[b] @ Rh[b].T)
                for b in range(self.B)])
            vh.append(v)
        vel_hist = torch.stack(vh, 1)

        a_ext = torch.zeros_like(parts)
        pt = torch.zeros(3, device=self.device)
        fc = torch.zeros(3, device=self.device)
        if act_body >= 0:
            pt = torch.tensor(act_point, dtype=torch.float32, device=self.device)
            fc = torch.tensor(act_force, dtype=torch.float32, device=self.device)
            sel = self.body_ids[:len(self.body_ids_np)] == act_body  # real nodes only
            feat = action_feature(
                parts[sel].detach().cpu().numpy(), pt.cpu().numpy(),
                fc.cpu().numpy(), float(self.mass[act_body]))
            a_ext[sel] = torch.tensor(feat, dtype=torch.float32, device=self.device)

        parts_np = parts.detach().cpu().numpy()
        s_np, r_np = build_edges(
            parts_np, vel_hist[:, -1].detach().cpu().numpy(), self.body_ids_np)
        ef_np = edge_features(parts_np, s_np, r_np, self.body_ids_np)
        E = len(s_np)
        E_pad = max(1, -(-E // EDGE_BUCKET)) * EDGE_BUCKET
        pad = np.full(E_pad - E, len(parts_np), np.int64)
        senders = torch.tensor(np.concatenate([s_np, pad]), device=self.device)
        receivers = torch.tensor(np.concatenate([r_np, pad]), device=self.device)
        ef = torch.tensor(np.concatenate(
            [ef_np, np.zeros((E_pad - E, ef_np.shape[1]), np.float32)]),
            device=self.device)

        zero3 = torch.zeros(self.n_pad, 3, device=self.device)
        batch = {
            "particles": torch.cat([parts, zero3]),
            "vel_hist": torch.cat(
                [vel_hist, torch.zeros(self.n_pad, C.HISTORY, 3, device=self.device)]),
            "body_ids": self.body_ids,
            "dist_ground": torch.cat(
                [parts[:, 2].clamp(0, C.CONTACT_RADIUS), zero3[:, 0]]),
            "a_ext": torch.cat([a_ext, zero3]),
            "mass": self.mass, "inertia": self.inertia,
        }
        pred = self.model(
            self.norm.node_features(batch), ef, senders, receivers,
            self.body_ids, self.B + 1, self.body_scalars)
        residual = self.norm.denorm_target(pred[:self.B])

        if self.free_flight:
            # Free-flight rule (mirrors web/public/js/sim.js): a body with no
            # edge to another body and no particle within the contact radius
            # of the floor feels only gravity and the applied force, so its
            # learned residual is zeroed. The network never saw a motionless
            # unsupported body in training and holds one in mid-air.
            touched = np.zeros(self.B, bool)
            bs, br = self.body_ids_np[s_np], self.body_ids_np[r_np]
            cross = bs != br
            touched[bs[cross]] = True
            touched[br[cross]] = True
            touched[np.unique(self.body_ids_np[parts_np[:, 2] < C.CONTACT_RADIUS])] = True
            if not touched.all():
                keep = torch.tensor(touched, device=self.device).float()[:, None]
                residual = residual * keep

        if self.energy_rule:
            residual = self._limit_energy(residual, act_body, pt, fc)

        ext_lin, ext_ang = external_accels(
            self.pos, self.quat, self.mass, self.inertia, act_body, pt, fc)
        self.pos, self.quat, linvel, angvel = step(
            self.pos, self.quat, self.lin_hist[-1], self.ang_hist[-1],
            residual, ext_lin, ext_ang)
        if self.ground_guard:
            R = quat_to_matrix(self.quat)
            for b in range(self.B):
                minz = float((self.offsets[b] @ R[b].T)[:, 2].min() + self.pos[b, 2])
                if minz < 0:
                    self.pos = self.pos.clone()
                    self.pos[b, 2] = self.pos[b, 2] - minz
                    if linvel[b, 2] < 0:
                        linvel = linvel.clone()
                        linvel[b, 2] = 0.0
        self.lin_hist = self.lin_hist[1:] + [linvel]
        self.ang_hist = self.ang_hist[1:] + [angvel]
        self.quat_hist = self.quat_hist[1:] + [self.quat]
        return self.pos.detach().cpu().numpy(), self.quat.detach().cpu().numpy()
