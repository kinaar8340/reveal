"""The three layers this harness measures.

Names are fixed. They are bookkeeping, not a claim that any of
9, 37, α, or W_g is global_R in reality.

    residual_R = observed − Π(global_R, frame_R)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Documented toe RubikConeConduit.braiding_target. Not measured here.
PHI_B_TARGET: float = 0.8145

# Mean within-class circular dispersion (rad) below which a geometric
# residual is called "small" for the name-change null.
RESIDUAL_SURVIVAL_THRESHOLD: float = 0.5

# Uniform-on-circle mean absolute deviation from a point. Used when a
# label class is too small to estimate a cluster. Finite; never inf.
NULL_CIRCULAR_DISPERSION: float = 0.5 * float(np.pi)


@dataclass(frozen=True)
class LayerNames:
    """Canonical layer identifiers. Keep these strings in CSV and code."""

    global_R: str = "global_R"
    frame_R: str = "frame_R"
    residual_R: str = "residual_R"
    local_pointer: str = "local_pointer"
    peloton: str = "peloton"


def residual_R(observed: np.ndarray, predicted: np.ndarray) -> float:
    """RMS of observed minus prediction from (global_R, frame_R)."""
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(f"shape mismatch: observed {obs.shape} vs predicted {pred.shape}")
    if obs.size == 0:
        return 0.0
    diff = obs - pred
    return float(np.sqrt(np.mean(diff * diff)))


def circular_dispersion(angles: np.ndarray) -> float:
    """Mean absolute circular deviation from the circular mean (radians)."""
    ang = np.asarray(angles, dtype=float).reshape(-1)
    if ang.size == 0:
        return float("nan")
    s = float(np.mean(np.sin(ang)))
    c = float(np.mean(np.cos(ang)))
    mu = float(np.arctan2(s, c))
    d = np.mod(ang - mu + np.pi, 2.0 * np.pi) - np.pi
    return float(np.mean(np.abs(d)))


def geometric_residual(
    labels: np.ndarray,
    angles: np.ndarray,
    *,
    min_count: int = 3,
) -> float:
    """Occupancy-weighted within-label circular dispersion of angles.

    Classes with fewer than ``min_count`` points cannot claim a cluster.
    They contribute :data:`NULL_CIRCULAR_DISPERSION` (π/2), the uniform-circle
    null — a large finite number, not infinity. Paired or unique stamps
    therefore report a large residual, not a discovery.

    Small ⇒ the stamp predicts geometry. Large ⇒ the stamp is a name.
    """
    labs = np.asarray(labels)
    ang = np.asarray(angles, dtype=float)
    if labs.size != ang.size or labs.size == 0:
        raise ValueError("labels and angles must be the same non-zero length")
    parts: list[float] = []
    weights: list[float] = []
    for lab in np.unique(labs):
        mask = labs == lab
        n = int(np.count_nonzero(mask))
        if n < min_count:
            parts.append(NULL_CIRCULAR_DISPERSION)
        else:
            parts.append(circular_dispersion(ang[mask]))
        weights.append(float(n))
    return float(np.average(parts, weights=weights))
