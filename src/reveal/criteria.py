"""Versioned local alignment tests for Experiment D.

v1 is leftover C: EXTERNAL, residual below the pack (floor 1.64 when
comparable), chain-paint heading rotation below C's fail of 0.81 rad,
window has a start and an end, obtained is always false.

v2 adds one harder clause only: the same candidate must pass the next
window with no refit. Criteria never roll backward in a run.

Never promote a mirror to GLOBAL. Never refit the attractor from this opening.
"""

from __future__ import annotations

from dataclasses import dataclass

from .leftover import CHAIN_AZ_THRESH

V1 = "v1"
V2 = "v2"

# Published C peloton residual, seed 0, 5000 steps, 96 sites.
PACK_FLOOR = 1.64

# 0.81 rad was a fail in C. Cap is C's chain threshold, not a kinder RMS.
CHAIN_ROTATION_CAP = CHAIN_AZ_THRESH

INELIGIBLE_NAMES = frozenset(
    {
        "peloton",
        "peloton_alpha",
        "alpha",
        "mean_theta",
        "theta_bar",
        "w_g",
        "wg",
        "lab_fitted",
        "site_mean",
        "no_external_input",
    }
)

TAG_EXTERNAL = "EXTERNAL"
TAG_SYNTHETIC = "SYNTHETIC_CONTROL"
TAG_INELIGIBLE = "INELIGIBLE"


@dataclass(frozen=True)
class Criteria:
    version: str = V1
    pack_floor: float = PACK_FLOOR
    chain_cap: float = CHAIN_ROTATION_CAP
    require_next_window: bool = False


def criteria_v1() -> Criteria:
    return Criteria(version=V1, require_next_window=False)


def bump(current: Criteria) -> Criteria:
    """Harden after a pass. Never decrease the version."""
    if current.version == V1:
        return Criteria(
            version=V2,
            pack_floor=current.pack_floor,
            chain_cap=current.chain_cap,
            require_next_window=True,
        )
    return current


def classify_tag(name: str, *, from_this_q: bool, synthetic: bool) -> str:
    key = name.strip().lower().replace(" ", "_")
    if from_this_q or key in INELIGIBLE_NAMES:
        return TAG_INELIGIBLE
    if synthetic:
        return TAG_SYNTHETIC
    return TAG_EXTERNAL


def residual_ok(residual_rms: float, peloton_rms: float, floor: float = PACK_FLOOR) -> bool:
    """Not worse than this window's pack, and below the published C floor.

    Matching the pack is not a win unless the tag is EXTERNAL/SYNTHETIC_CONTROL
    and the heading was not computed from this q. Ineligible pack still fails
    on the tag. A heading at ~π/2 cannot sneak under the 1.64 floor alone.
    """
    return bool(residual_rms <= peloton_rms + 1e-12 and residual_rms < floor)


def evaluate_v1(
    *,
    tag: str,
    residual_rms: float,
    peloton_rms: float,
    chain_rotation: float,
    window_start: int | None,
    window_end: int | None,
    criteria: Criteria | None = None,
) -> bool:
    """A window pass is a mirror interval, not north. obtained stays false."""
    spec = criteria or criteria_v1()
    if tag == TAG_INELIGIBLE:
        return False
    if window_start is None or window_end is None or int(window_end) <= int(window_start):
        return False
    if not residual_ok(residual_rms, peloton_rms, spec.pack_floor):
        return False
    if chain_rotation >= spec.chain_cap:
        return False
    return True
