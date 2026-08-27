"""Experiment B — name-change null on the 9/π orbit.

Fixed geometry: step 9/π, method step_index.
Labels: mod9, mod37, paired(9,37), shuffled.
angle_bin runs as CONTROL, never EVIDENCE.

Lock is not claimed unless a geometric residual stays small across
mod 9 and mod 37 under step_index. step_index exNMI is never advertised
as lock. Do not add moduli.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from .layers import RESIDUAL_SURVIVAL_THRESHOLD, geometric_residual
from .paths import import_vortex_core

TINY_STEPS = 40
FULL_STEPS = 400
TINY_PERMUTATIONS = 8
FULL_PERMUTATIONS = 48
SHUFFLE_SEED = 0

# The only moduli this experiment is allowed to stamp with.
ALLOWED_MODULI = (9, 37)


@dataclass
class NameRow:
    method: str
    modulus: int
    control: bool
    steps: int
    exNMI: float
    nmi: float
    nmi_z: float
    residual_R: float
    survives_name_change: bool
    lock_claimed: bool

    def as_row(self) -> dict[str, object]:
        return {
            "method": self.method,
            "modulus": self.modulus,
            "control": self.control,
            "steps": self.steps,
            "exNMI": self.exNMI,
            "nmi": self.nmi,
            "nmi_z": self.nmi_z,
            "residual_R": self.residual_R,
            "survives_name_change": self.survives_name_change,
            "lock_claimed": self.lock_claimed,
        }


NAMES_CSV_FIELDS = [
    "method",
    "modulus",
    "control",
    "steps",
    "exNMI",
    "nmi",
    "nmi_z",
    "residual_R",
    "survives_name_change",
    "lock_claimed",
]


def _core() -> ModuleType:
    return import_vortex_core()


def _align(
    core: ModuleType,
    labels: np.ndarray,
    angles: np.ndarray,
    n_permutations: int,
) -> dict:
    return core.label_angle_alignment(
        labels,
        angles,
        n_angle_bins=max(9, 18),
        n_permutations=n_permutations,
        rng=np.random.default_rng(SHUFFLE_SEED),
    )


def _row(
    *,
    method: str,
    modulus: int,
    control: bool,
    steps: int,
    labels: np.ndarray,
    angles: np.ndarray,
    n_permutations: int,
    core: ModuleType,
    survives: bool,
) -> NameRow:
    if modulus not in ALLOWED_MODULI:
        raise ValueError(f"modulus {modulus} is not in {ALLOWED_MODULI}; do not add moduli")
    align = _align(core, labels, angles, n_permutations)
    residual = geometric_residual(labels, angles)
    lock_claimed = bool(
        survives and (not control) and method == "step_index"
    )
    return NameRow(
        method=method,
        modulus=int(modulus),
        control=bool(control),
        steps=int(steps),
        exNMI=float(align["nmi_excess"]),
        nmi=float(align["nmi"]),
        nmi_z=float(align["nmi_z"]),
        residual_R=float(residual),
        survives_name_change=bool(survives) if method == "step_index" else False,
        lock_claimed=lock_claimed,
    )


def run_names(
    *,
    steps: int = TINY_STEPS,
    n_permutations: int = TINY_PERMUTATIONS,
    core: ModuleType | None = None,
) -> list[NameRow]:
    """Stamp the same 9/π orbit with four names plus the angle_bin CONTROL."""
    if steps < 2:
        raise ValueError("steps must be >= 2")
    vm = core or _core()
    step = float(vm.DEFAULT_STEP_RADIANS)
    angles = vm.circle_angles(steps, step)

    labels_m9 = vm.labels_for_orbit(
        steps, step_radians=step, method="step_index", modulus=9
    )
    labels_m37 = vm.labels_for_orbit(
        steps, step_radians=step, method="step_index", modulus=37
    )
    labels_paired = vm.labels_for_orbit(
        steps, step_radians=step, method="paired", modulus=37
    )
    labels_bin = vm.labels_for_orbit(
        steps, step_radians=step, method="angle_bin", modulus=9
    )
    shuffled = np.array(labels_m9, copy=True)
    np.random.default_rng(SHUFFLE_SEED).shuffle(shuffled)

    res9 = geometric_residual(labels_m9, angles)
    res37 = geometric_residual(labels_m37, angles)
    survives = bool(
        res9 < RESIDUAL_SURVIVAL_THRESHOLD and res37 < RESIDUAL_SURVIVAL_THRESHOLD
    )

    rows = [
        _row(
            method="step_index",
            modulus=9,
            control=False,
            steps=steps,
            labels=labels_m9,
            angles=angles,
            n_permutations=n_permutations,
            core=vm,
            survives=survives,
        ),
        _row(
            method="step_index",
            modulus=37,
            control=False,
            steps=steps,
            labels=labels_m37,
            angles=angles,
            n_permutations=n_permutations,
            core=vm,
            survives=survives,
        ),
        _row(
            method="paired",
            modulus=37,
            control=False,
            steps=steps,
            labels=labels_paired,
            angles=angles,
            n_permutations=n_permutations,
            core=vm,
            survives=survives,
        ),
        _row(
            method="shuffled",
            modulus=9,
            control=False,
            steps=steps,
            labels=shuffled,
            angles=angles,
            n_permutations=n_permutations,
            core=vm,
            survives=survives,
        ),
        _row(
            method="angle_bin",
            modulus=9,
            control=True,
            steps=steps,
            labels=labels_bin,
            angles=angles,
            n_permutations=n_permutations,
            core=vm,
            survives=survives,
        ),
    ]
    return rows


def write_names_csv(rows: list[NameRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=NAMES_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row())
    return path
