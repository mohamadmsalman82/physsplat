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
IMPULSE_STEPS = 3        # steps over which a user poke is applied
