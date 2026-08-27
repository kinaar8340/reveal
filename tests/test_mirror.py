"""Experiment D: idle default, ineligible pack, synthetic control, versioning."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reveal.criteria import (
    TAG_INELIGIBLE,
    TAG_SYNTHETIC,
    V1,
    V2,
    bump,
    classify_tag,
    criteria_v1,
)
from reveal.mirror import run_mirror


def _aligned(steps: int, n_sites: int, axis: list[float]) -> np.ndarray:
    vec = np.array(axis, dtype=float)
    vec = vec / np.linalg.norm(vec)
    return np.broadcast_to(vec, (steps, n_sites, 3)).copy()


def test_no_file_is_idle_no_mirror():
    result = run_mirror(steps=8, seed=0, n_sites=4)
    assert result.no_mirror is True
    assert result.obtained is False
    assert result.event == "idle"
    assert result.candidate == "no_external_input"


def test_ineligible_peloton_cannot_pass():
    ptrs = _aligned(8, 4, [1.0, 0.0, 0.0])
    result = run_mirror(
        steps=8,
        seed=0,
        n_sites=4,
        heading=np.array([1.0, 0.0, 0.0]),
        from_this_q=True,
        candidate_name="peloton",
        pointers_t=ptrs,
    )
    assert all(w.tag == TAG_INELIGIBLE for w in result.windows)
    assert all(w.mirror is False for w in result.windows)
    assert all(w.obtained is False for w in result.windows)
    assert result.no_mirror is True


def test_synthetic_that_is_the_pack_fails():
    ptrs = _aligned(8, 4, [0.0, 1.0, 0.0])
    pack = np.array([0.0, 1.0, 0.0])
    result = run_mirror(
        steps=8,
        seed=0,
        n_sites=4,
        heading=pack,
        from_this_q=True,
        synthetic=True,
        candidate_name="synthetic",
        pointers_t=ptrs,
    )
    assert result.tag == TAG_INELIGIBLE
    assert all(w.mirror is False for w in result.windows)


def test_independent_synthetic_may_pass_v1_then_v2_fails_on_rotate():
    ptrs = _aligned(8, 4, [1.0, 0.0, 0.0])
    result = run_mirror(
        steps=8,
        seed=0,
        n_sites=4,
        heading=np.array([1.0, 0.0, 0.0]),
        from_this_q=False,
        synthetic=True,
        candidate_name="synthetic",
        pointers_t=ptrs,
        rotate_after_first_pass=True,
    )
    assert result.tag == TAG_SYNTHETIC
    assert result.windows[0].mirror is True
    assert result.windows[0].criteria_ver == V1
    assert result.windows[0].obtained is False
    later = result.windows[1:]
    assert any(w.event == "mirror_ended" for w in later)
    assert all(w.obtained is False for w in result.windows)
    assert result.obtained is False


def test_criteria_version_only_increases():
    ptrs = _aligned(8, 4, [1.0, 0.0, 0.0])
    result = run_mirror(
        steps=8,
        seed=1,
        n_sites=4,
        heading=np.array([1.0, 0.0, 0.0]),
        synthetic=True,
        pointers_t=ptrs,
    )
    order = {"v1": 1, "v2": 2}
    versions = [order[w.criteria_ver] for w in result.windows]
    assert versions == sorted(versions)
    bumped = bump(criteria_v1())
    assert bumped.version == V2
    assert bump(bumped).version == V2


def test_angle_bin_and_extra_moduli_never_appear():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src/reveal/mirror.py").read_text() + (root / "src/reveal/criteria.py").read_text()
    assert "angle_bin" not in text
    assert "111" not in text
    assert "333" not in text


def test_classify_peloton_ineligible():
    assert classify_tag("peloton", from_this_q=False, synthetic=False) == TAG_INELIGIBLE
    assert classify_tag("W_g", from_this_q=False, synthetic=False) == TAG_INELIGIBLE
    assert classify_tag("file", from_this_q=False, synthetic=False) == "EXTERNAL"
