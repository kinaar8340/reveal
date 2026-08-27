"""CLI: python -m reveal header|names|all."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .header import (
    FULL_SITES,
    FULL_STEPS as HEADER_FULL_STEPS,
    HeaderIdentityError,
    TINY_SITES,
    TINY_STEPS as HEADER_TINY_STEPS,
    run_ablation,
    write_header_csv,
    write_header_trace,
)
from .leftover import run_leftover, write_leftover_csv
from .names import (
    FULL_PERMUTATIONS,
    FULL_STEPS as NAMES_FULL_STEPS,
    TINY_PERMUTATIONS,
    TINY_STEPS as NAMES_TINY_STEPS,
    run_names,
    write_names_csv,
)
from .paths import default_outputs
from .plots import (
    plot_header_burst_rms,
    plot_header_mean_theta,
    plot_header_phi_b,
    plot_leftover_residuals,
    plot_names_exnmi,
)


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=None, help="Override default step count")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full", action="store_true", help="Longer defaults (still no videos)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/outputs)",
    )


def _out_dir(ns: argparse.Namespace) -> Path:
    path = ns.out if ns.out is not None else default_outputs()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _header_params(ns: argparse.Namespace) -> tuple[int, int]:
    if ns.full:
        steps = HEADER_FULL_STEPS if ns.steps is None else ns.steps
        sites = FULL_SITES
    else:
        steps = HEADER_TINY_STEPS if ns.steps is None else ns.steps
        sites = TINY_SITES
    return steps, sites


def _names_params(ns: argparse.Namespace) -> tuple[int, int]:
    if ns.full:
        steps = NAMES_FULL_STEPS if ns.steps is None else ns.steps
        perms = FULL_PERMUTATIONS
    else:
        steps = NAMES_TINY_STEPS if ns.steps is None else ns.steps
        perms = TINY_PERMUTATIONS
    return steps, perms


def cmd_header(ns: argparse.Namespace) -> int:
    steps, sites = _header_params(ns)
    out = _out_dir(ns)
    try:
        headed, unheaded = run_ablation(steps=steps, seed=ns.seed, n_sites=sites)
    except HeaderIdentityError as exc:
        print(f"header job failed: {exc}", file=sys.stderr)
        return 1
    write_header_csv([headed, unheaded], out / "header.csv")
    write_header_trace(headed, unheaded, out / "header_trace.csv")
    plot_header_burst_rms(headed, unheaded, out / "header_burst_rms.png")
    plot_header_mean_theta(headed, unheaded, out / "header_mean_theta.png")
    plot_header_phi_b(headed, unheaded, out / "header_phi_b.png")
    for arm in (headed, unheaded):
        tag = "headed" if arm.headed else "unheaded"
        first_pi = "" if arm.first_mean_pi is None else arm.first_mean_pi
        first_tcrit = "" if arm.first_mean_tcrit is None else arm.first_mean_tcrit
        first_burst = "" if arm.first_burst_step is None else arm.first_burst_step
        print(
            f"header {tag} bursts={arm.burst_count} "
            f"first_burst={first_burst} "
            f"mean_Theta={arm.mean_Theta:.4f} "
            f"mean_Theta_late={arm.mean_Theta_late:.4f} "
            f"rms_dTheta={arm.rms_dTheta:.4f} "
            f"rms_dTheta_late={arm.rms_dTheta_late:.4f} "
            f"first_pi={first_pi} first_tcrit={first_tcrit}"
        )
    print(f"wrote {out / 'header.csv'}")
    return 0


def cmd_names(ns: argparse.Namespace) -> int:
    steps, perms = _names_params(ns)
    out = _out_dir(ns)
    rows = run_names(steps=steps, n_permutations=perms)
    write_names_csv(rows, out / "names.csv")
    plot_names_exnmi(rows, out / "names_exnmi.png")
    for row in rows:
        tag = "CONTROL" if row.control else "probe"
        print(
            f"names method={row.method} m={row.modulus} {tag} "
            f"exNMI={row.exNMI:.4f} residual_R={row.residual_R:.4f} "
            f"lock_claimed={row.lock_claimed}"
        )
    print(f"wrote {out / 'names.csv'}")
    return 0


def cmd_leftover(ns: argparse.Namespace) -> int:
    steps, sites = _header_params(ns)
    out = _out_dir(ns)
    rows = run_leftover(steps=steps, seed=ns.seed, n_sites=sites)
    write_leftover_csv(rows, out / "leftover.csv")
    plot_leftover_residuals(rows, out / "leftover_residuals.png")
    for row in rows:
        tag = "PELOTON" if row.peloton else "probe"
        print(
            f"leftover candidate={row.candidate} {tag} "
            f"residual_rms={row.residual_rms:.4f} "
            f"beats_peloton={row.beats_peloton} "
            f"survives_name_change={row.survives_name_change} "
            f"lock_claimed={row.lock_claimed}"
        )
    print(f"wrote {out / 'leftover.csv'}")
    return 0


def cmd_all(ns: argparse.Namespace) -> int:
    rc = cmd_header(ns)
    if rc != 0:
        return rc
    rc = cmd_names(ns)
    if rc != 0:
        return rc
    return cmd_leftover(ns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reveal",
        description="Measure residuals. Does not name God.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_header = sub.add_parser("header", help="two-gyro header on/off")
    _add_shared(p_header)
    p_header.set_defaults(func=cmd_header)
    p_names = sub.add_parser("names", help="9/π name-change null")
    _add_shared(p_names)
    p_names.set_defaults(func=cmd_names)
    p_leftover = sub.add_parser("leftover", help="local_pointer leftover-lock search")
    _add_shared(p_leftover)
    p_leftover.set_defaults(func=cmd_leftover)
    p_all = sub.add_parser("all", help="header then names then leftover")
    _add_shared(p_all)
    p_all.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return int(ns.func(ns))
