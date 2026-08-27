"""Experiment D — windowed search for an external attractor.

A pass is a mirror interval, not north. obtained is always False.
Default with no off-opening input is no_mirror (idle).
Synthetic headings are SYNTHETIC_CONTROL, not discovery.
Do not refit the attractor from this opening. Do not promote a mirror to GLOBAL.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from flux_hopf_lib.constants import theta_crit
from flux_hopf_lib.flux.lattice import FluxLatticeConfig, gauge_restoring_alpha
from flux_hopf_lib.quaternion.core import q_conj, q_mult, q_normalize, small_rotor
from numpy.typing import NDArray

from .criteria import (
    TAG_INELIGIBLE,
    TAG_SYNTHETIC,
    V1,
    bump,
    classify_tag,
    criteria_v1,
    evaluate_v1,
)
from .header import TINY_SITES, TINY_STEPS, _burst_reconnect, _twist
from .leftover import (
    local_pointers,
    paint_chain,
    peloton_vector,
    residual_rms,
    wrap_delta,
    xy_heading,
)


@dataclass
class WindowRow:
    window_id: int | str
    start: int | str
    end: int | str
    candidate: str
    tag: str
    residual_rms: float | str
    peloton_rms: float | str
    beats_peloton: bool
    chain_rotation: float | str
    mirror: bool
    obtained: bool
    criteria_ver: str
    no_mirror: bool
    event: str = ""

    def as_row(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "start": self.start,
            "end": self.end,
            "candidate": self.candidate,
            "tag": self.tag,
            "residual_rms": self.residual_rms,
            "peloton_rms": self.peloton_rms,
            "beats_peloton": self.beats_peloton,
            "chain_rotation": self.chain_rotation,
            "mirror": self.mirror,
            "obtained": self.obtained,
            "criteria_ver": self.criteria_ver,
            "no_mirror": self.no_mirror,
            "event": self.event,
        }


WINDOW_CSV_FIELDS = [
    "window_id",
    "start",
    "end",
    "candidate",
    "tag",
    "residual_rms",
    "peloton_rms",
    "beats_peloton",
    "chain_rotation",
    "mirror",
    "obtained",
    "criteria_ver",
    "no_mirror",
    "event",
]

SUMMARY_CSV_FIELDS = [
    "no_mirror",
    "obtained",
    "criteria_ver",
    "n_windows",
    "n_mirror_windows",
    "candidate",
    "tag",
    "event",
    "steps",
    "seed",
    "n_sites",
]


@dataclass
class MirrorResult:
    no_mirror: bool
    obtained: bool
    criteria_ver: str
    windows: list[WindowRow]
    candidate: str
    tag: str
    event: str
    steps: int
    seed: int
    n_sites: int
    peloton_trace: list[float] = field(default_factory=list)
    residual_trace: list[float] = field(default_factory=list)
    window_centers: list[float] = field(default_factory=list)

    def summary_row(self) -> dict[str, object]:
        return {
            "no_mirror": self.no_mirror,
            "obtained": False,
            "criteria_ver": self.criteria_ver,
            "n_windows": len(self.windows),
            "n_mirror_windows": sum(1 for w in self.windows if w.mirror),
            "candidate": self.candidate,
            "tag": self.tag,
            "event": self.event,
            "steps": self.steps,
            "seed": self.seed,
            "n_sites": self.n_sites,
        }


def idle_result(*, steps: int = 0, seed: int = 0, n_sites: int = 0) -> MirrorResult:
    row = WindowRow(
        window_id="",
        start="",
        end="",
        candidate="no_external_input",
        tag=TAG_INELIGIBLE,
        residual_rms="",
        peloton_rms="",
        beats_peloton=False,
        chain_rotation="",
        mirror=False,
        obtained=False,
        criteria_ver=V1,
        no_mirror=True,
        event="idle",
    )
    return MirrorResult(
        no_mirror=True,
        obtained=False,
        criteria_ver=V1,
        windows=[row],
        candidate="no_external_input",
        tag=TAG_INELIGIBLE,
        event="idle",
        steps=steps,
        seed=seed,
        n_sites=n_sites,
    )


def load_attractor_csv(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load t,x,y,z unit-vector series not produced by this process."""
    times: list[float] = []
    vecs: list[list[float]] = []
    with Path(path).open() as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("attractor csv needs a header t,x,y,z")
        fields = {name.strip().lower() for name in reader.fieldnames}
        if not {"t", "x", "y", "z"} <= fields:
            raise ValueError("attractor csv must have columns t,x,y,z")
        for raw in reader:
            key = {k.strip().lower(): v for k, v in raw.items()}
            times.append(float(key["t"]))
            vec = np.array([float(key["x"]), float(key["y"]), float(key["z"])], dtype=float)
            n = float(np.linalg.norm(vec))
            if n < 1e-12:
                raise ValueError("attractor row has a zero vector")
            vecs.append((vec / n).tolist())
    if not times:
        raise ValueError("attractor csv is empty")
    return np.asarray(times, dtype=float), np.asarray(vecs, dtype=float)


def heading_at(times: NDArray[np.floating], vecs: NDArray[np.floating], t: float) -> NDArray[np.float64]:
    if times.size == 1:
        return np.array(vecs[0], dtype=float)
    if t <= times[0]:
        return np.array(vecs[0], dtype=float)
    if t >= times[-1]:
        return np.array(vecs[-1], dtype=float)
    i = int(np.searchsorted(times, t, side="right") - 1)
    i = min(max(i, 0), times.size - 2)
    span = float(times[i + 1] - times[i])
    w = 0.0 if span <= 0 else (t - float(times[i])) / span
    mixed = (1.0 - w) * vecs[i] + w * vecs[i + 1]
    n = float(np.linalg.norm(mixed))
    return mixed / n if n > 1e-12 else np.array(vecs[i], dtype=float)


def synthetic_heading(seed: int) -> NDArray[np.float64]:
    """Independent RNG heading. seed must not be the lattice seed."""
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=3)
    return vec / float(np.linalg.norm(vec))


def _record_unheaded_pointers(
    *,
    steps: int,
    seed: int,
    n_sites: int,
) -> NDArray[np.float64]:
    """Unheaded two-gyro, same primitives as header. Pointers each step."""
    cfg = FluxLatticeConfig()
    t_crit = float(cfg.theta_crit if cfg.theta_crit is not None else theta_crit(cfg.kappa))
    rng = np.random.default_rng(seed)
    q = q_normalize(rng.standard_normal((n_sites, 4)))
    delta_L = small_rotor(cfg.omega_L)
    delta_R = small_rotor(cfg.omega_R)
    delta_R_inv = q_conj(delta_R)
    out = np.zeros((steps, n_sites, 3), dtype=float)
    for step in range(steps):
        q = q_normalize(q_mult(q_mult(delta_L, q), delta_R_inv))
        twist = _twist(q)
        avg = float(np.mean(twist) % (2.0 * np.pi))
        gauge_restoring_alpha(avg, gauge_strength=cfg.gauge_strength, kappa=cfg.kappa)
        q, twist, _n = _burst_reconnect(q, twist, t_crit)
        out[step] = local_pointers(q)
    return out


def _windows(steps: int) -> list[tuple[int, int]]:
    n_win = 2 if steps < 16 else min(8, max(2, steps // 4))
    edges = np.linspace(0, steps, n_win + 1, dtype=int)
    spans = []
    for i in range(n_win):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            b = min(steps, a + 1)
        spans.append((a, b))
    return spans


def _chain_rotation(pointers: NDArray[np.floating]) -> float:
    az9, _r9 = xy_heading(paint_chain(pointers, 9))
    az37, _r37 = xy_heading(paint_chain(pointers, 37))
    return wrap_delta(az9, az37)


def run_windows(
    pointers_t: NDArray[np.floating],
    heading_fn,
    *,
    candidate: str,
    tag: str,
    rotate_after_first_pass: bool = False,
) -> MirrorResult:
    """Evaluate heading_fn(window_end_step, window_id) on each window."""
    steps, n_sites, _dim = pointers_t.shape
    spans = _windows(steps)
    spec = criteria_v1()
    rows: list[WindowRow] = []
    residual_trace: list[float] = []
    peloton_trace: list[float] = []
    centers: list[float] = []
    event = ""
    promoting = True
    passed_once = False
    final_ver = V1

    for i, (start, end) in enumerate(spans):
        chunk = pointers_t[start:end].reshape(-1, 3)
        pack = peloton_vector(chunk)
        peloton_rms = residual_rms(chunk, pack)
        heading = np.asarray(heading_fn(end - 1, i), dtype=float)
        if rotate_after_first_pass and passed_once:
            heading = np.array([-heading[1], heading[0], heading[2]], dtype=float)
            n = float(np.linalg.norm(heading))
            heading = heading / n
        rms = residual_rms(chunk, heading)
        if tag == TAG_INELIGIBLE:
            chain_rot = _chain_rotation(chunk)
        else:
            # Candidate is not refit from this opening. Paint cannot rotate it.
            chain_rot = 0.0
        beats = bool(rms < peloton_rms)
        used_ver = spec.version
        mirror = False
        this_event = ""
        if promoting:
            mirror = evaluate_v1(
                tag=tag,
                residual_rms=rms,
                peloton_rms=peloton_rms,
                chain_rotation=chain_rot,
                window_start=start,
                window_end=end,
                criteria=spec,
            )
            if spec.require_next_window and not mirror and passed_once:
                this_event = "mirror_ended"
                event = "mirror_ended"
                promoting = False
            elif mirror:
                passed_once = True
                spec = bump(spec)
                final_ver = spec.version
        rows.append(
            WindowRow(
                window_id=i,
                start=start,
                end=end,
                candidate=candidate,
                tag=tag,
                residual_rms=rms,
                peloton_rms=peloton_rms,
                beats_peloton=beats,
                chain_rotation=chain_rot,
                mirror=mirror,
                obtained=False,
                criteria_ver=used_ver,
                no_mirror=False,
                event=this_event,
            )
        )
        residual_trace.append(rms)
        peloton_trace.append(peloton_rms)
        centers.append(0.5 * (start + end))

    any_mirror = any(w.mirror for w in rows)
    if not any_mirror:
        final_ver = V1

    return MirrorResult(
        no_mirror=not any_mirror,
        obtained=False,
        criteria_ver=final_ver,
        windows=rows,
        candidate=candidate,
        tag=tag,
        event=event,
        steps=steps,
        seed=0,
        n_sites=n_sites,
        peloton_trace=peloton_trace,
        residual_trace=residual_trace,
        window_centers=centers,
    )


def run_mirror(
    *,
    steps: int = TINY_STEPS,
    seed: int = 0,
    n_sites: int = TINY_SITES,
    attractor: Path | None = None,
    synthetic: bool = False,
    synthetic_seed: int | None = None,
    heading: NDArray[np.floating] | None = None,
    from_this_q: bool = False,
    candidate_name: str | None = None,
    pointers_t: NDArray[np.floating] | None = None,
    rotate_after_first_pass: bool = False,
) -> MirrorResult:
    """Idle unless an off-opening heading is supplied."""
    if attractor is None and not synthetic and heading is None:
        return idle_result(steps=steps, seed=seed, n_sites=n_sites)

    syn_seed = seed + 7919 if synthetic_seed is None else int(synthetic_seed)
    if syn_seed == seed and synthetic:
        syn_seed = seed + 7919

    file_times: NDArray[np.float64] | None = None
    file_vecs: NDArray[np.float64] | None = None
    if attractor is not None:
        file_times, file_vecs = load_attractor_csv(Path(attractor))
        name = candidate_name or Path(attractor).name
        synthetic_flag = False
    elif heading is not None:
        name = candidate_name or ("peloton" if from_this_q else "synthetic")
        synthetic_flag = synthetic and not from_this_q
    else:
        heading = synthetic_heading(syn_seed)
        name = candidate_name or "synthetic"
        synthetic_flag = True

    if pointers_t is None:
        pointers_t = _record_unheaded_pointers(steps=steps, seed=seed, n_sites=n_sites)

    tag = classify_tag(name, from_this_q=from_this_q, synthetic=synthetic_flag)

    if from_this_q and heading is None:
        chunk0 = pointers_t[0]
        heading = peloton_vector(chunk0.reshape(-1, 3) if chunk0.ndim == 3 else chunk0)
        name = candidate_name or "peloton"
        tag = TAG_INELIGIBLE

    def heading_fn(t: int, _i: int) -> NDArray[np.float64]:
        if file_times is not None and file_vecs is not None:
            return heading_at(file_times, file_vecs, float(t))
        return np.asarray(heading, dtype=float)

    result = run_windows(
        pointers_t,
        heading_fn,
        candidate=name,
        tag=tag,
        rotate_after_first_pass=rotate_after_first_pass,
    )
    result.steps = steps
    result.seed = seed
    result.n_sites = n_sites
    result.obtained = False
    return result


def write_mirror_csv(result: MirrorResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(result.summary_row())
    return path


def write_mirror_windows_csv(result: MirrorResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=WINDOW_CSV_FIELDS)
        writer.writeheader()
        for row in result.windows:
            writer.writerow(row.as_row())
    return path
