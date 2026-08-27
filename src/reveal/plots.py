"""Three plots: burst/δΘ RMS, pointer-α stability, exNMI by label method."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from .header import HeaderRun
from .names import NameRow


def plot_header_burst_rms(
    headed: HeaderRun,
    unheaded: HeaderRun,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["burst_count", "rms_δΘ"]
    headed_vals = [headed.burst_count, headed.rms_dTheta]
    unheaded_vals = [unheaded.burst_count, unheaded.rms_dTheta]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, headed_vals, width, label="headed")
    ax.bar(x + width / 2, unheaded_vals, width, label="unheaded")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("value")
    ax.set_title("Header ablation — burst count and δΘ RMS")
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
    """Pointer α stability. Not a φ_b lock claim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(headed.pointer_history, label="headed pointer")
    ax.plot(unheaded.pointer_history, label="unheaded pointer", linestyle="--")
    ax.set_xlabel("step")
    ax.set_ylabel("tanh(α · 6)")
    ax.set_title("Pointer / α stability (not a φ_b lock claim)")
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
