"""Plots: burst/δΘ RMS, mean Θ, pointer α, exNMI by label method."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from .header import HeaderRun
from .leftover import LeftoverRow
from .mirror import MirrorResult
from .names import NameRow


def plot_header_burst_rms(
    headed: HeaderRun,
    unheaded: HeaderRun,
    path: Path,
) -> Path:
    """Two panels so burst counts do not hide the RMS scale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 0.35
    x = np.array([0.0, 1.0])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))

    ax = axes[0]
    ax.bar(x[0] - width / 2, headed.burst_count, width, label="headed")
    ax.bar(x[0] + width / 2, unheaded.burst_count, width, label="unheaded")
    ax.set_xticks([0.0])
    ax.set_xticklabels(["burst_count"])
    ax.set_ylabel("count")
    ax.set_title("Bursts")
    ax.legend()

    ax = axes[1]
    ax.bar(
        x - width / 2,
        [headed.rms_dTheta, headed.rms_dTheta_late],
        width,
        label="headed",
    )
    ax.bar(
        x + width / 2,
        [unheaded.rms_dTheta, unheaded.rms_dTheta_late],
        width,
        label="unheaded",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["rms_δΘ", "rms_δΘ late"])
    ax.set_ylabel("rad")
    ax.set_title("δΘ RMS")
    ax.legend()

    fig.suptitle("Header ablation — burst count and δΘ RMS")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_header_mean_theta(
    headed: HeaderRun,
    unheaded: HeaderRun,
    path: Path,
) -> Path:
    """Spatial-mean Θ vs step. The hold-down question lives here, not on tanh(6α)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(headed.mean_theta_history, label="headed")
    ax.plot(unheaded.mean_theta_history, label="unheaded", linestyle="--")
    ax.axhline(np.pi, color="gray", linewidth=0.8, label="π")
    ax.axhline(headed.theta_crit, color="black", linewidth=0.8, linestyle=":", label="θ_crit")
    if headed.burn_in_steps:
        ax.axvline(headed.burn_in_steps, color="0.6", linewidth=0.8, linestyle="--")
    ax.set_xlabel("step")
    ax.set_ylabel("mean Θ (rad)")
    ax.set_title("Header ablation — mean Θ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_header_phi_b(
    headed: HeaderRun,
    unheaded: HeaderRun,
    path: Path,
) -> Path:
    """Raw α, not tanh(6α). Saturated tanh is not a φ_b lock claim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(headed.alpha_history, label="headed α")
    ax.plot(unheaded.alpha_history, label="unheaded α", linestyle="--")
    ax.set_xlabel("step")
    ax.set_ylabel("α")
    ax.set_title("Pointer / α (not a φ_b lock claim)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_names_exnmi(rows: list[NameRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{r.method}\nm={r.modulus}" for r in rows]
    values = [r.exNMI for r in rows]
    colors = ["#c44e52" if r.control else "#4c72b0" for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(range(len(rows)), values, color=colors)
    for bar, row in zip(bars, rows):
        if row.control:
            bar.set_hatch("//")
            bar.set_edgecolor("black")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("exNMI")
    ax.set_title("exNMI by label method — angle_bin is CONTROL")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_leftover_residuals(rows: list[LeftoverRow], path: Path) -> Path:
    """Residual RMS vs candidate. Peloton is hatched, never painted as north."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r.candidate for r in rows]
    values = [r.residual_rms for r in rows]
    colors = ["#c44e52" if r.peloton else "#4c72b0" for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(range(len(rows)), values, color=colors)
    for bar, row in zip(bars, rows):
        if row.peloton:
            bar.set_hatch("//")
            bar.set_edgecolor("black")
    if rows:
        ax.axhline(
            rows[0].peloton_residual_rms,
            color="gray",
            linewidth=0.8,
            label="peloton residual",
        )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("residual RMS (rad)")
    ax.set_title("Leftover lock — peloton is PELOTON, not north")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_mirror_residuals(result: MirrorResult, path: Path) -> Path:
    """Window residual vs pack. Pack is hatched / baseline, never north."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    if result.residual_trace:
        x = result.window_centers
        ax.plot(x, result.residual_trace, label=result.candidate)
        ax.plot(x, result.peloton_trace, linestyle="--", label="peloton residual")
        ax.fill_between(x, result.peloton_trace, alpha=0.15, hatch="//", label="pack")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no_mirror (idle)", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("step")
    ax.set_ylabel("residual RMS (rad)")
    ax.set_title("Mirror windows — pack hatched, obtained=False")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
