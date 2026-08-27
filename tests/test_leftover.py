"""Leftover lock: peloton is not north; lab axes must beat the pack to claim lock."""

from __future__ import annotations

from pathlib import Path

from reveal.leftover import (
    CHAIN_MODULI,
    LEFTOVER_CSV_FIELDS,
    chain_labels,
    run_leftover,
    write_leftover_csv,
)


def test_peloton_is_not_claimed_as_north():
    rows = run_leftover(steps=4, seed=0, n_sites=4)
    by_name = {r.candidate: r for r in rows}
    assert "peloton" in by_name
    assert "peloton_alpha" in by_name
    assert by_name["peloton"].peloton is True
    assert by_name["peloton"].independent is False
    assert by_name["peloton"].lock_claimed is False
    assert by_name["peloton_alpha"].peloton is True
    assert by_name["peloton_alpha"].lock_claimed is False
    assert by_name["peloton"].beats_peloton is False


def test_lab_candidates_are_independent():
    rows = run_leftover(steps=4, seed=1, n_sites=4)
    labs = [r for r in rows if r.candidate.startswith("lab_")]
    assert {r.candidate for r in labs} == {"lab_x", "lab_y", "lab_z"}
    assert all(r.independent is True for r in labs)
    assert all(r.peloton is False for r in labs)
    assert "W_g" not in {r.candidate for r in rows}
    assert "wg" not in {r.candidate.lower() for r in rows}


def test_other_seed_is_independent_not_this_mean():
    rows = run_leftover(steps=4, seed=0, n_sites=4)
    other = [r for r in rows if r.candidate == "other_seed"]
    assert len(other) == 1
    assert other[0].independent is True
    assert other[0].peloton is False


def test_lock_claimed_false_on_tiny_random_init():
    rows = run_leftover(steps=8, seed=0, n_sites=8)
    assert all(r.lock_claimed is False for r in rows)


def test_chain_moduli_are_only_nine_and_thirty_seven():
    assert CHAIN_MODULI == (9, 37)
    labels9 = chain_labels(10, 9)
    labels37 = chain_labels(10, 37)
    assert labels9.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 0]
    assert labels37.tolist() == list(range(10))


def test_write_leftover_csv(tmp_path: Path):
    rows = run_leftover(steps=4, seed=0, n_sites=4)
    path = write_leftover_csv(rows, tmp_path / "leftover.csv")
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines[0].split(",") == LEFTOVER_CSV_FIELDS
    assert "peloton" in text
    assert "GLOBAL" not in text
    assert "north" not in text.lower()
