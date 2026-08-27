"""Leftover-lock search: local_pointer vs a heading that is not this run's mean.

Bike map (names only; not a claim that any of them is north):
  wheel+tube  — continuous field (q_i, the thing that goes around)
  valve       — W_g = 350/π, fill port. Not heading.
  chain       — modulus 9 / 37. Transmission period. Not heading.
  handlebar   — frame_R, local square
  pointer     — local_pointer, forward bolted to this bar
  peloton     — α / mean of the bars. Shareable. Still made of the bikes.
  road / north — global_R, if it exists. Not cut from this bar.

A candidate is global_R only if residuals beat the peloton residual, the
candidate is independent of this run's Θ̄, and the fitted heading does not
rotate when the chain is swapped. Otherwise the bike is a local machine
with no north. That is an allowed outcome.

Do not search by renaming Θ̄. Do not use W_g as a heading. Do not add moduli.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flux_hopf_lib.quaternion.core import q_conj, q_mult, q_normalize
from numpy.typing import NDArray

from .header import TINY_SITES, TINY_STEPS, run_header
from .paths import import_vortex_core

FORWARD = np.array([1.0, 0.0, 0.0], dtype=float)
CHAIN_MODULI = (9, 37)
RESULTANT_THRESH = 0.5
CHAIN_AZ_THRESH = 0.3


@dataclass
class LeftoverRow:
    candidate: str
    independent: bool
    peloton: bool
    residual_rms: float
    peloton_residual_rms: float
    beats_peloton: bool
    fit_az_m9: float
    fit_az_m37: float
    fit_resultant_m9: float
    fit_resultant_m37: float
    chain_delta_az: float
    survives_name_change: bool
    lock_claimed: bool
    steps: int
    seed: int
    n_sites: int

    def as_row(self) -> dict[str, object]:
        return {
            "candidate": self.candidate,
            "independent": self.independent,
            "peloton": self.peloton,
            "residual_rms": self.residual_rms,
            "peloton_residual_rms": self.peloton_residual_rms,
            "beats_peloton": self.beats_peloton,
            "fit_az_m9": self.fit_az_m9,
            "fit_az_m37": self.fit_az_m37,
            "fit_resultant_m9": self.fit_resultant_m9,
            "fit_resultant_m37": self.fit_resultant_m37,
            "chain_delta_az": self.chain_delta_az,
            "survives_name_change": self.survives_name_change,
            "lock_claimed": self.lock_claimed,
            "steps": self.steps,
            "seed": self.seed,
            "n_sites": self.n_sites,
        }


LEFTOVER_CSV_FIELDS = [
    "candidate",
    "independent",
    "peloton",
    "residual_rms",
    "peloton_residual_rms",
    "beats_peloton",
    "fit_az_m9",
    "fit_az_m37",
    "fit_resultant_m9",
    "fit_resultant_m37",
    "chain_delta_az",
    "survives_name_change",
    "lock_claimed",
    "steps",
    "seed",
    "n_sites",
]


def q_rotate(q: NDArray[np.floating], v: NDArray[np.floating]) -> NDArray[np.float64]:
    """Rotate vector(s) v by quaternion(s) q: q v q*."""
    q = q_normalize(np.asarray(q, dtype=float))
    v = np.asarray(v, dtype=float)
    squeeze = q.ndim == 1
    if squeeze:
        q = q[None, :]
    n = q.shape[0]
    if v.ndim == 1:
        v = np.broadcast_to(v, (n, 3))
    zeros = np.zeros((n, 1))
    vq = np.concatenate([zeros, v], axis=-1)
    out = q_mult(q_mult(q, vq), q_conj(q))[..., 1:]
    return out[0] if squeeze else out


def _unit(v: NDArray[np.floating], axis: int = -1) -> NDArray[np.float64]:
    arr = np.asarray(v, dtype=float)
    n = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.maximum(n, 1e-12)


def vector_angle(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.float64]:
    aa = _unit(a)
    bb = _unit(b)
    dots = np.sum(aa * bb, axis=-1)
    return np.arccos(np.clip(dots, -1.0, 1.0))


def peloton_vector(pointers: NDArray[np.floating]) -> NDArray[np.float64]:
    mean = np.mean(np.asarray(pointers, dtype=float), axis=0)
    return _unit(mean)


def residual_rms(pointers: NDArray[np.floating], candidate: NDArray[np.floating]) -> float:
    return float(np.sqrt(np.mean(vector_angle(pointers, candidate) ** 2)))


def rotate_about_z(vectors: NDArray[np.floating], phi: NDArray[np.floating]) -> NDArray[np.float64]:
    v = np.asarray(vectors, dtype=float)
    p = np.asarray(phi, dtype=float)
    c = np.cos(p)
    s = np.sin(p)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.stack([c * x - s * y, s * x + c * y, z], axis=-1)


def wrap_delta(a: float, b: float) -> float:
    return float(abs((a - b + np.pi) % (2.0 * np.pi) - np.pi))


def xy_heading(vectors: NDArray[np.floating]) -> tuple[float, float]:
    """Azimuth of the mean xy projection, and that projection's resultant length."""
    v = np.asarray(vectors, dtype=float)
    mx = float(np.mean(v[:, 0]))
    my = float(np.mean(v[:, 1]))
    resultant = float(np.hypot(mx, my))
    return float(np.arctan2(my, mx)), resultant


def chain_labels(n_sites: int, modulus: int) -> NDArray[np.int_]:
    if modulus not in CHAIN_MODULI:
        raise ValueError(f"modulus {modulus} is not in {CHAIN_MODULI}; do not add moduli")
    core = import_vortex_core()
    return np.array([core.modular_label(i, modulus) for i in range(n_sites)], dtype=int)


def paint_chain(pointers: NDArray[np.floating], modulus: int) -> NDArray[np.float64]:
    """Rotate each pointer about z by 2π · (site index mod m) / m. Paint, not heading."""
    n = np.asarray(pointers).shape[0]
    labels = chain_labels(n, modulus)
    phi = 2.0 * np.pi * labels.astype(float) / float(modulus)
    return rotate_about_z(pointers, phi)


def local_pointers(q: NDArray[np.floating]) -> NDArray[np.float64]:
    """Forward axis bolted to each bar: R(q) ê_x, perpendicular to local z."""
    return q_rotate(q, FORWARD)


def _survives(az9: float, r9: float, az37: float, r37: float) -> bool:
    if r9 < RESULTANT_THRESH or r37 < RESULTANT_THRESH:
        return False
    return wrap_delta(az9, az37) < CHAIN_AZ_THRESH


def _row(
    *,
    candidate: str,
    independent: bool,
    peloton: bool,
    pointers: NDArray[np.float64],
    heading: NDArray[np.float64],
    peloton_rms: float,
    painted9: NDArray[np.float64],
    painted37: NDArray[np.float64],
    survives: bool,
    steps: int,
    seed: int,
    n_sites: int,
) -> LeftoverRow:
    rms = residual_rms(pointers, heading)
    beats = bool(rms < peloton_rms)
    az9, r9 = xy_heading(painted9)
    az37, r37 = xy_heading(painted37)
    lock_claimed = bool(independent and beats and survives)
    return LeftoverRow(
        candidate=candidate,
        independent=independent,
        peloton=peloton,
        residual_rms=rms,
        peloton_residual_rms=peloton_rms,
        beats_peloton=beats,
        fit_az_m9=az9,
        fit_az_m37=az37,
        fit_resultant_m9=r9,
        fit_resultant_m37=r37,
        chain_delta_az=wrap_delta(az9, az37),
        survives_name_change=survives,
        lock_claimed=lock_claimed,
        steps=steps,
        seed=seed,
        n_sites=n_sites,
    )


def run_leftover(
    *,
    steps: int = TINY_STEPS,
    seed: int = 0,
    n_sites: int = TINY_SITES,
) -> list[LeftoverRow]:
    """Unheaded lattice only. The header already failed as a header."""
    run = run_header(headed=False, steps=steps, seed=seed, n_sites=n_sites)
    other = run_header(headed=False, steps=steps, seed=seed + 1, n_sites=n_sites)
    pointers = local_pointers(run.final_q)
    other_pointers = local_pointers(other.final_q)
    pack = peloton_vector(pointers)
    other_pack = peloton_vector(other_pointers)
    peloton_rms = residual_rms(pointers, pack)
    painted9 = paint_chain(pointers, 9)
    painted37 = paint_chain(pointers, 37)
    az9, r9 = xy_heading(painted9)
    az37, r37 = xy_heading(painted37)
    pack_survives = _survives(az9, r9, az37, r37)
    alpha_rot = q_rotate(
        np.array([np.cos(run.mean_alpha), 0.0, 0.0, np.sin(run.mean_alpha)], dtype=float),
        FORWARD,
    )
    candidates: list[tuple[str, NDArray[np.float64], bool, bool]] = [
        ("peloton", pack, False, True),
        ("peloton_alpha", alpha_rot, False, True),
        ("lab_x", np.array([1.0, 0.0, 0.0]), True, False),
        ("lab_y", np.array([0.0, 1.0, 0.0]), True, False),
        ("lab_z", np.array([0.0, 0.0, 1.0]), True, False),
        ("other_seed", other_pack, True, False),
    ]
    return [
        _row(
            candidate=name,
            independent=independent,
            peloton=is_pack,
            pointers=pointers,
            heading=heading,
            peloton_rms=peloton_rms,
            painted9=painted9,
            painted37=painted37,
            survives=pack_survives if is_pack else True,
            steps=steps,
            seed=seed,
            n_sites=n_sites,
        )
        for name, heading, independent, is_pack in candidates
    ]


def write_leftover_csv(rows: list[LeftoverRow], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEFTOVER_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row())
    return path
