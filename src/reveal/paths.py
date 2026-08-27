"""Sibling-repo resolution. reveal sits next to toe and vortex_math."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


def repo_root() -> Path:
    """reveal repository root (parent of src/)."""
    return Path(__file__).resolve().parents[2]


def projects_root() -> Path:
    return repo_root().parent


def vortex_math_root() -> Path:
    return projects_root() / "vortex_math"


def toe_root() -> Path:
    return projects_root() / "toe"


def flux_hopf_lib_root() -> Path:
    return projects_root() / "flux_hopf_lib"


def default_outputs() -> Path:
    return repo_root() / "outputs"


def import_vortex_core() -> ModuleType:
    """Import vortex_math's src.core without requiring it to be pip-installed.

    vortex_math has no pyproject; its tests do ``sys.path.insert(root); import src.core``.
    """
    root = vortex_math_root()
    if not root.is_dir():
        raise ImportError(
            f"vortex_math not found at {root}. "
            "reveal expects to sit next to ../vortex_math."
        )
    marker = str(root)
    if marker not in sys.path:
        sys.path.insert(0, marker)
    import src.core as core  # noqa: PLC0415  — sibling adapter

    return core
