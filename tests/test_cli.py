"""CLI smoke: tiny steps, no theology, required artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from reveal.cli import main

THEOLOGY = ("Matthew", "Christian", "atheist", "18:20")


def test_header_cli_two_rows(tmp_path: Path, capsys):
    rc = main(["header", "--steps", "4", "--seed", "0", "--out", str(tmp_path)])
    assert rc == 0
    csv_path = tmp_path / "header.csv"
    assert csv_path.is_file()
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    headed_flags = {row["headed"] for row in rows}
    assert headed_flags == {"True", "False"}
    assert (tmp_path / "header_burst_rms.png").is_file()
    assert (tmp_path / "header_phi_b.png").is_file()
    out = capsys.readouterr().out
    for word in THEOLOGY:
        assert word not in out
    assert "LOCKED" not in out


def test_names_cli_control_and_no_lock(tmp_path: Path, capsys):
    rc = main(["names", "--steps", "48", "--out", str(tmp_path)])
    assert rc == 0
    csv_path = tmp_path / "names.csv"
    assert csv_path.is_file()
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    assert "step_index" in by_method
    assert all(r["lock_claimed"] == "False" for r in by_method["step_index"])
    angle = by_method["angle_bin"][0]
    assert angle["control"] == "True"
    assert angle["lock_claimed"] == "False"
    assert (tmp_path / "names_exnmi.png").is_file()
    out = capsys.readouterr().out
    for word in THEOLOGY:
        assert word not in out
    assert "LOCKED" not in out
    assert "CONTROL" in out
    assert "EVIDENCE" not in out


def test_all_writes_both(tmp_path: Path):
    rc = main(["all", "--steps", "48", "--seed", "0", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "header.csv").is_file()
    assert (tmp_path / "names.csv").is_file()
    assert (tmp_path / "header_burst_rms.png").is_file()
    assert (tmp_path / "header_phi_b.png").is_file()
    assert (tmp_path / "names_exnmi.png").is_file()
