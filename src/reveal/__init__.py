"""reveal — residual harness on toe two-gyro and vortex_math 9/π.

residual_R = observed − Π(global_R, frame_R)

This package measures residuals. It does not name God.
"""

from .layers import (
    NULL_CIRCULAR_DISPERSION,
    PHI_B_TARGET,
    RESIDUAL_SURVIVAL_THRESHOLD,
    LayerNames,
    residual_R,
)

__version__ = "0.1.0"

__all__ = [
    "NULL_CIRCULAR_DISPERSION",
    "PHI_B_TARGET",
    "RESIDUAL_SURVIVAL_THRESHOLD",
    "LayerNames",
    "residual_R",
    "__version__",
]
