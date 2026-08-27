"""Name-change null: step_index is not a lock; angle_bin is CONTROL."""

from __future__ import annotations

from pathlib import Path

from reveal.names import ALLOWED_MODULI, NAMES_CSV_FIELDS, run_names, write_names_csv


def test_step_index_does_not_claim_lock():
    rows = run_names(steps=48, n_permutations=8)
    step = [r for r in rows if r.method == "step_index"]
    assert len(step) == 2
    assert {r.modulus for r in step} == {9, 37}
    assert all(r.lock_claimed is False for r in step)
    assert all(r.control is False for r in step)
    assert all(r.survives_name_change is False for r in step)


def test_angle_bin_is_control_not_evidence():
    rows = run_names(steps=48, n_permutations=8)
    bins = [r for r in rows if r.method == "angle_bin"]
    assert len(bins) == 1
    row = bins[0]
    assert row.control is True
    assert row.lock_claimed is False


def test_shuffled_row_exists():
    rows = run_names(steps=48, n_permutations=8)
    shuffled = [r for r in rows if r.method == "shuffled"]
    assert len(shuffled) == 1
    assert shuffled[0].control is False
    assert shuffled[0].lock_claimed is False


def test_paired_row_exists():
    rows = run_names(steps=48, n_permutations=8)
    paired = [r for r in rows if r.method == "paired"]
    assert len(paired) == 1
    assert paired[0].modulus == 37
    assert paired[0].lock_claimed is False


def test_no_extra_moduli():
    rows = run_names(steps=48, n_permutations=8)
    assert {r.modulus for r in rows} <= set(ALLOWED_MODULI)
    assert len(rows) == 5


def test_write_names_csv(tmp_path: Path):
    rows = run_names(steps=48, n_permutations=8)
    path = write_names_csv(rows, tmp_path / "names.csv")
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines[0].split(",") == NAMES_CSV_FIELDS
    assert len(lines) == 6
    assert "CONTROL" not in text  # flag is boolean control, not a slogan in the CSV
    assert "angle_bin" in text
