"""Semi-implicit Euler on SE(3), pure torch functions (differentiable, and
ported line-by-line to TypeScript in Phase 7).

Everything analytically knowable is integrated exactly: gravity, and the
user/scripted external force whose vector AND application point are known
(F/m linear; I^-1 (r x F) angular). The network contributes only the
residual: contact and friction. This guarantees point-of-application
physics by construction: off-center grabs lean, end grabs dangle, releases
free-fall exactly (design doc, integration section).
"""

import torch

from ..common.constants import DT, GRAVITY


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion product a*b, batched (..., 4)."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], -1)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    """Rotation vector (..., 3) -> xyzw quaternion, small-angle safe."""
    angle = v.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    small = angle < 1e-8
    k = torch.where(small, 0.5 - angle**2 / 48, torch.sin(half) / angle.clamp_min(1e-12))
    return torch.cat([v * k, torch.cos(half)], -1)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    row = lambda *c: torch.stack(c, -1)
    return torch.stack([
        row(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        row(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        row(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    ], -2)


def external_accels(
    pos: torch.Tensor, quat: torch.Tensor, mass: torch.Tensor,
    inertia: torch.Tensor, act_body: int, act_point: torch.Tensor,
    act_force: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic Newton-Euler contribution of the known external force.
    Returns (B,3) linear and (B,3) angular acceleration, zero except the
    acted body."""
    B = pos.shape[0]
    lin = torch.zeros_like(pos)
    ang = torch.zeros_like(pos)
    if act_body >= 0:
        lin[act_body] = act_force / mass[act_body]
        R = quat_to_matrix(quat[act_body])
        I_w = R @ torch.diag(inertia[act_body]) @ R.T
        torque = torch.linalg.cross(act_point - pos[act_body], act_force)
        ang[act_body] = torch.linalg.solve(I_w, torque)
    return lin, ang


def step(
    pos: torch.Tensor,      # (B, 3)
    quat: torch.Tensor,     # (B, 4) xyzw
    linvel: torch.Tensor,   # (B, 3)
    angvel: torch.Tensor,   # (B, 3)
    residual: torch.Tensor, # (B, 6) DENORMALIZED network output (accel)
    ext_lin: torch.Tensor,  # (B, 3) analytic external linear accel
    ext_ang: torch.Tensor,  # (B, 3)
    dt: float = DT,
):
    """One semi-implicit Euler step. Returns new (pos, quat, linvel, angvel)."""
    g = torch.tensor([0.0, 0.0, -GRAVITY], device=pos.device, dtype=pos.dtype)
    linvel = linvel + (g + ext_lin + residual[:, :3]) * dt
    angvel = angvel + (ext_ang + residual[:, 3:]) * dt
    pos = pos + linvel * dt
    quat = quat_mul(quat_from_rotvec(angvel * dt), quat)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return pos, quat, linvel, angvel


def kabsch(world: torch.Tensor, offsets: torch.Tensor):
    """Best-fit rigid transform mapping body-frame offsets onto predicted
    world points (fallback head only). Returns (R (3,3), t (3,))."""
    c_w = world.mean(0)
    c_r = offsets.mean(0)
    A = (world - c_w).T @ (offsets - c_r)   # (3, 3) cross-covariance
    U, _, Vt = torch.linalg.svd(A)
    d = torch.det(U @ Vt)
    S = torch.diag(torch.tensor([1.0, 1.0, d], device=world.device))
    R = U @ S @ Vt
    return R, c_w - R @ c_r
