"""Physics invariants: the project's independent oracles.

Every check here derives from physics, never from the code under test, so
they cannot be gamed by a buggy implementation. Used three ways:
1. as unit-test assertions,
2. as rejection filters during data generation (a trajectory violating them
   never enters the dataset),
3. as runtime metrics on every model rollout (Phase 4 scorecard).
"""

from .invariants import (
    energy,
    max_speed,
    min_height,
    rigidity_error,
    rotation_validity_error,
    within_bounds,
)

__all__ = [
    "rigidity_error",
    "rotation_validity_error",
    "energy",
    "max_speed",
    "min_height",
    "within_bounds",
]
