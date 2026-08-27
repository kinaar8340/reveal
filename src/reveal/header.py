"""Experiment A — header ablation on the two-gyro conduit.

headed:   two-gyro THEN q ← q · g(α) THEN burst
unheaded: two-gyro THEN burst

Same Δω, κ, θ_crit, seed, steps. Wrap flux_hopf_lib quaternion + gauge
primitives. Burst reconnection is the four-line rule from
toe/scripts/two_gyro_lattice_demo.py. Not the 3-torus PDE.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from flux_hopf_lib.constants import W_G_LOCK, theta_crit
from flux_hopf_lib.flux.lattice import (
    FluxLatticeConfig,
    gauge_restoring_alpha,
    pointer_damping,
)
from flux_hopf_lib.quaternion.core import q_conj, q_mult, q_normalize, small_rotor
from numpy.typing import NDArray

from .layers import PHI_B_TARGET, residual_R as rms_residual

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

TINY_STEPS = 8
FULL_STEPS = 200
TINY_SITES = 8
FULL_SITES = 96


class HeaderIdentityError(RuntimeError):
    """Headed and unheaded traces matched while the gauge path was supposed to be on."""


def _opt_int(value: int | None) -> str:
    return "" if value is None else str(value)


def _first_at_or_above(series: list[float], threshold: float) -> int | None:
    for i, value in enumerate(series):
        if value >= threshold:
            return i
    return None


def _burn_in_steps(n_steps: int) -> int:
    """Drop the first 20% on long runs; keep the whole window when N is tiny."""
    if n_steps < 20:
        return 0
    return max(1, n_steps // 5)


@dataclass
class HeaderRun:
    headed: bool
    seed: int
    steps: int
    n_sites: int
    kappa: float
    omega_L: float
    omega_R: float
    theta_crit: float
    burst_count: int
    mean_Theta: float
    mean_Theta_late: float
    rms_dTheta: float
    rms_dTheta_late: float
    residual_R: float
    mean_alpha: float
    pointer_rms: float
    burn_in_steps: int
    first_mean_pi: int | None
    first_mean_tcrit: int | None
    first_burst_step: int | None
    wg_target: float
    wg_lock: bool
    phi_b_target: float
    phi_b_lock: bool
    final_q: NDArray[np.float64] = field(repr=False)
    pointer_history: list[float] = field(repr=False)
    mean_theta_history: list[float] = field(repr=False)
    alpha_history: list[float] = field(repr=False)
    rms_history: list[float] = field(repr=False)
    burst_history: list[int] = field(repr=False)

    def as_row(self) -> dict[str, object]:
        return {
            "headed": self.headed,
            "seed": self.seed,
            "steps": self.steps,
            "n_sites": self.n_sites,
            "kappa": self.kappa,
            "omega_L": self.omega_L,
            "omega_R": self.omega_R,
            "theta_crit": self.theta_crit,
            "burst_count": self.burst_count,
            "mean_Theta": self.mean_Theta,
            "mean_Theta_late": self.mean_Theta_late,
            "rms_dTheta": self.rms_dTheta,
            "rms_dTheta_late": self.rms_dTheta_late,
            "residual_R": self.residual_R,
            "mean_alpha": self.mean_alpha,
            "pointer_rms": self.pointer_rms,
            "burn_in_steps": self.burn_in_steps,
            "first_mean_pi": _opt_int(self.first_mean_pi),
            "first_mean_tcrit": _opt_int(self.first_mean_tcrit),
            "first_burst_step": _opt_int(self.first_burst_step),
            "wg_target": self.wg_target,
            "wg_lock": self.wg_lock,
            "phi_b_target": self.phi_b_target,
            "phi_b_lock": self.phi_b_lock,
        }


HEADER_CSV_FIELDS = [
    "headed",
    "seed",
    "steps",
    "n_sites",
    "kappa",
    "omega_L",
    "omega_R",
    "theta_crit",
    "burst_count",
    "mean_Theta",
    "mean_Theta_late",
    "rms_dTheta",
    "rms_dTheta_late",
    "residual_R",
    "mean_alpha",
    "pointer_rms",
    "burn_in_steps",
    "first_mean_pi",
    "first_mean_tcrit",
    "first_burst_step",
    "wg_target",
    "wg_lock",
    "phi_b_target",
    "phi_b_lock",
]

TRACE_CSV_FIELDS = [
    "step",
    "headed_mean_Theta",
    "unheaded_mean_Theta",
    "headed_rms",
    "unheaded_rms",
    "headed_bursts",
    "unheaded_bursts",
]


def _twist(q: NDArray[np.floating]) -> NDArray[np.float64]:
    return 2.0 * np.arccos(np.clip(np.asarray(q, dtype=float)[..., 0], -1.0, 1.0))


def _burst_reconnect(
    q: NDArray[np.float64],
    twist: NDArray[np.float64],
    t_crit: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Toe two-gyro demo reconnection. Not the PDE sink."""
    mask = twist > t_crit
    n_burst = int(np.count_nonzero(mask))
    if n_burst == 0:
        return q, twist, 0
    out_q = np.array(q, copy=True, dtype=float)
    out_t = np.array(twist, copy=True, dtype=float)
    mixed = 0.3 * IDENTITY + 0.7 * out_q[mask]
    out_q[mask] = q_normalize(mixed)
    out_t[mask] *= 0.15
    return out_q, out_t, n_burst


def run_header(
    *,
    headed: bool,
    steps: int = TINY_STEPS,
    seed: int = 0,
    n_sites: int = TINY_SITES,
    config: FluxLatticeConfig | None = None,
) -> HeaderRun:
    """One two-gyro lattice trajectory with header on or off."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")

    cfg = config or FluxLatticeConfig()
    t_crit = float(cfg.theta_crit if cfg.theta_crit is not None else theta_crit(cfg.kappa))
    rng = np.random.default_rng(seed)
    q = q_normalize(rng.standard_normal((n_sites, 4)))
    twist = _twist(q)

    delta_L = small_rotor(cfg.omega_L)
    delta_R = small_rotor(cfg.omega_R)
    delta_R_inv = q_conj(delta_R)

    burst_count = 0
    pred_chunks: list[NDArray[np.float64]] = []
    obs_chunks: list[NDArray[np.float64]] = []
    pointer_history: list[float] = []
    mean_theta_history: list[float] = []
    alpha_history: list[float] = []
    spatial_rms: list[float] = []
    burst_history: list[int] = []

    for _ in range(steps):
        # frame_R: two-gyro  q ← δL q δR†
        q = q_normalize(q_mult(q_mult(delta_L, q), delta_R_inv))
        twist = _twist(q)

        avg = float(np.mean(twist) % (2.0 * np.pi))
        alpha = gauge_restoring_alpha(
            avg, gauge_strength=cfg.gauge_strength, kappa=cfg.kappa
        )
        alpha_history.append(alpha)
        pointer_history.append(pointer_damping(alpha))

        # global_R applied only when headed: q ← q · g(α)
        if headed:
            gauge_rot = np.array([np.cos(alpha), 0.0, 0.0, np.sin(alpha)], dtype=float)
            q = q_normalize(q_mult(q, gauge_rot))
            twist = _twist(q)

        pred = np.array(twist, copy=True, dtype=float)
        q, twist, n_burst = _burst_reconnect(q, twist, t_crit)
        burst_count += n_burst
        burst_history.append(n_burst)
        pred_chunks.append(pred)
        obs_chunks.append(np.array(twist, copy=True, dtype=float))

        mean_t = float(np.mean(twist))
        mean_theta_history.append(mean_t)
        spatial_rms.append(float(np.sqrt(np.mean((twist - mean_t) ** 2))))

    obs = np.concatenate(obs_chunks)
    pred = np.concatenate(pred_chunks)
    residual = rms_residual(obs, pred)
    rms_dtheta = float(np.mean(spatial_rms)) if spatial_rms else 0.0
    pointer = np.asarray(pointer_history, dtype=float)
    burn = _burn_in_steps(steps)
    late_rms = spatial_rms[burn:] or spatial_rms
    late_mean = mean_theta_history[burn:] or mean_theta_history

    return HeaderRun(
        headed=headed,
        seed=seed,
        steps=steps,
        n_sites=n_sites,
        kappa=float(cfg.kappa),
        omega_L=float(cfg.omega_L),
        omega_R=float(cfg.omega_R),
        theta_crit=t_crit,
        burst_count=burst_count,
        mean_Theta=float(np.mean(mean_theta_history)),
        mean_Theta_late=float(np.mean(late_mean)),
        rms_dTheta=rms_dtheta,
        rms_dTheta_late=float(np.mean(late_rms)),
        residual_R=residual,
        mean_alpha=float(np.mean(alpha_history)),
        pointer_rms=float(np.sqrt(np.mean(pointer * pointer))) if pointer.size else 0.0,
        burn_in_steps=burn,
        first_mean_pi=_first_at_or_above(mean_theta_history, float(np.pi)),
        first_mean_tcrit=_first_at_or_above(mean_theta_history, t_crit),
        first_burst_step=_first_at_or_above([float(n) for n in burst_history], 1.0),
        wg_target=float(W_G_LOCK),
        wg_lock=False,
        phi_b_target=float(PHI_B_TARGET),
        phi_b_lock=False,
        final_q=np.array(q, copy=True, dtype=float),
        pointer_history=pointer_history,
        mean_theta_history=mean_theta_history,
        alpha_history=alpha_history,
        rms_history=spatial_rms,
        burst_history=burst_history,
    )


def assert_header_changed(headed: HeaderRun, unheaded: HeaderRun) -> None:
    """Fail the job if headed ≡ unheaded while the gauge path is supposed to be on."""
    if headed.steps != unheaded.steps or headed.seed != unheaded.seed:
        raise ValueError("headed/unheaded comparison requires the same seed and steps")
    if not headed.headed or unheaded.headed:
        raise ValueError("expected headed=True and unheaded=False")
    mean_abs_alpha = float(np.mean(np.abs(headed.alpha_history))) if headed.alpha_history else 0.0
    if mean_abs_alpha <= 0.0:
        return
    if np.allclose(headed.final_q, unheaded.final_q, rtol=0.0, atol=1e-15):
        raise HeaderIdentityError(
            "headed and unheaded final quaternions are bit-identical; "
            "the gauge path was supposed to be on"
        )


def run_ablation(
    *,
    steps: int = TINY_STEPS,
    seed: int = 0,
    n_sites: int = TINY_SITES,
    config: FluxLatticeConfig | None = None,
) -> tuple[HeaderRun, HeaderRun]:
    headed = run_header(
        headed=True, steps=steps, seed=seed, n_sites=n_sites, config=config
    )
    unheaded = run_header(
        headed=False, steps=steps, seed=seed, n_sites=n_sites, config=config
    )
    assert_header_changed(headed, unheaded)
    return headed, unheaded


def write_header_csv(rows: list[HeaderRun], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row())
    return path


def write_header_trace(headed: HeaderRun, unheaded: HeaderRun, path: Path) -> Path:
    if headed.steps != unheaded.steps:
        raise ValueError("trace requires equal step counts")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRACE_CSV_FIELDS)
        writer.writeheader()
        for i in range(headed.steps):
            writer.writerow(
                {
                    "step": i,
                    "headed_mean_Theta": headed.mean_theta_history[i],
                    "unheaded_mean_Theta": unheaded.mean_theta_history[i],
                    "headed_rms": headed.rms_history[i],
                    "unheaded_rms": unheaded.rms_history[i],
                    "headed_bursts": headed.burst_history[i],
                    "unheaded_bursts": unheaded.burst_history[i],
                }
            )
    return path
