"""Single source of truth for every constant shared across phases.

Many hard-to-find bugs in this project are two files silently disagreeing on
one of these numbers (e.g. the data generator and the browser integrator using
different timesteps). Import from here; never redefine locally.
"""

# --- Time ----------------------------------------------------------------
DT = 1.0 / 60.0          # recorded/simulated timestep seen by the model (s)
INTERNAL_HZ = 240        # PyBullet internal substep rate; record every 4th step
RECORD_STEPS = 300       # frames recorded per trajectory (5 s at 60 Hz)
SETTLE_STEPS = 100       # settle period before recording starts

# --- Physics -------------------------------------------------------------
GRAVITY = 9.81           # m/s^2, acting along -z (z is up everywhere)
# Randomized per body at datagen time. Range covers hollow plastic pencils
# (~530 kg/m^3 effective, see docs/objects.md) through solid wood/plastic.
# Mass and inertia are model input features, so the network handles the range.
DENSITY_RANGE = (400.0, 900.0)   # kg/m^3
FRICTION_RANGE = (0.2, 0.9)      # smooth plastic barrel .. rubber grip
# Hard plastic on a hard desk bounces visibly; excluding dead-thud materials
# keeps the learned bounce crisp instead of averaged toward mush.
RESTITUTION_RANGE = (0.1, 0.5)

# --- Particles / graph ---------------------------------------------------
PARTICLE_SPACING = 0.004                 # target surface-sample spacing (m)
CONTACT_RADIUS = 1.5 * PARTICLE_SPACING  # edge radius for the GNN graph (m)
MAX_PARTICLES_PER_BODY = 400             # cap for very large objects
HISTORY = 5                              # velocity-history length fed to model

# --- Interaction ---------------------------------------------------------
IMPULSE_STEPS = 3        # recorded steps over which a user poke is applied
POKE_DELTA_V = (0.05, 0.30)   # poke strength as target velocity change (m/s)

# Grab = spring-damper from grab point to mouse/scripted target. Gains scale
# with body mass so a pencil and a heavy box feel the same: kp = m*omega^2,
# kd = 2*zeta*m*omega. Force capped at GRAB_FORCE_CAP * m * g.
GRAB_OMEGA = 12.6        # rad/s (~2 Hz spring)
GRAB_ZETA = 0.9          # near-critical damping -> dangles settle, not ring
GRAB_FORCE_CAP = 3.0     # x (m*g)
# Keep relative speeds inside the contact-sensing (tunneling) budget:
# CONTACT_RADIUS / DT = 0.36 m/s. Predictive edges extend it; caps respect it.
TARGET_SPEED_MAX = 0.25  # m/s, max grab-target speed

# --- Scene bounds (rejection filters) ------------------------------------
SCENE_XY_MAX = 0.8       # m; a body beyond this has escaped -> reject
SCENE_Z_MAX = 0.5        # m
# Penetration filter: a one-frame spike during an impact is real compliance;
# chronic overlap is bad ground truth. Reject on sustained depth (95th
# percentile over frames) or an extreme spike.
PENETRATION_SUSTAINED = 0.0015   # m, 95th percentile across frames
PENETRATION_SPIKE = 0.005        # m, absolute max
SPEED_REJECT = 3.0       # m/s; faster than this means the sim exploded
