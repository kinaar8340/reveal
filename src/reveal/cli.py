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
)
from .names import (
    FULL_PERMUTATIONS,
    FULL_STEPS as NAMES_FULL_STEPS,
    TINY_PERMUTATIONS,
    TINY_STEPS as NAMES_TINY_STEPS,
    run_names,
    write_names_csv,
)
from .paths import default_outputs
from .plots import plot_header_burst_rms, plot_header_phi_b, plot_names_exnmi


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
    plot_header_burst_rms(headed, unheaded, out / "header_burst_rms.png")
    plot_header_phi_b(headed, unheaded, out / "header_phi_b.png")
    print(
        f"header rows=2 headed_bursts={headed.burst_count} "
        f"unheaded_bursts={unheaded.burst_count} "
        f"headed_rms_dTheta={headed.rms_dTheta:.4f} "
        f"unheaded_rms_dTheta={unheaded.rms_dTheta:.4f}"
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


def cmd_all(ns: argparse.Namespace) -> int:
    rc = cmd_header(ns)
    if rc != 0:
        return rc
    return cmd_names(ns)


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
    p_all = sub.add_parser("all", help="header then names")
    _add_shared(p_all)
    p_all.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return int(ns.func(ns))
