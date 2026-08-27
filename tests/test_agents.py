"""AGENTS.md stays on the rails."""

from __future__ import annotations

from pathlib import Path

from reveal.paths import repo_root

REQUIRED = [
    "This repo measures residuals. It does not name God.",
    "header on/off is the only lattice comparison that matters",
    'modulus swap is a null for "is this just a name?"',
    "angle_bin is CONTROL",
    "do not add moduli to chase a lock",
    "do not add axes or seeds to beat the peloton residual",
    "do not commit large videos",
    "α and Θ̄ are peloton, not global_R",
    "W_g is the valve, not heading",
    "Never rename the mean of the bars as north.",
]


def test_agents_md_exists_with_rails():
    path = repo_root() / "AGENTS.md"
    assert path.is_file()
    text = path.read_text()
    for line in REQUIRED:
        assert line in text
