# reveal

This repo measures residuals. 

```
residual_R = observed − Π(global_R, frame_R)
```

- **global_R** — candidate pointer. Lattice: `α` from the existing two-gyro gauge. Circle: the fixed `9/π` geometry.
- **frame_R** — local assembly. Lattice: `q ← δL q δR†`. Circle: labeling method + modulus.
- **residual_R** — observed minus that prediction.

Two questions only:

1. Does the header change the residual?
2. Does any leftover survive changing the name?

`9`, `37`, `α`, and `W_g` are stamps and candidates in this harness. They are not declared to be `global_R` in reality.

## Install

```bash
cd ~/Projects/reveal
python -m venv .venv
source .venv/bin/activate
pip install -e ../flux_hopf_lib
pip install -e ".[dev]"
```

`vortex_math` is imported from `../vortex_math` (no extra install).

## Experiments

**A — header ablation (failed prediction).** Same `Δω`, `κ`, `θ_crit`, seed, steps. `headed` applies `q ← q · g(α)` after the two-gyro update; `unheaded` does not. Same burst reconnection as the toe two-gyro demo. The job fails if the two runs are bit-identical while the gauge path is supposed to be on. `W_g` / `φ_b` lock columns stay false: this harness does not spin up the torch conduit. Seed 0, 5000 steps, 96 sites: the prediction was fewer or later bursts headed, lower late `rms_δΘ` headed, mean twist held down. Published numbers are bursts 8 vs 9, first burst at step 4 vs 1574 (opposite), late `rms_δΘ` 1.119 vs 1.113 (tie or opposite), both arms sitting on π, spatial mean never reaching `θ_crit`. `g(α)` changes the path — headed `α` chatters, unheaded `α` wanders — but that is not alignment or load-bearing. A headed flywheel is not structurally superior in this code. This run opens on π at step 0–1 from random unit quaternions, not the paper’s climb from `Θ̄ ≈ 0.82`; that explains a clean fail and does not reverse it. Header-under-paper-init would be a new experiment, not this result undone.

**B — name-change null (holds).** Fixed geometry `9/π`, method `step_index`. Stamps: mod 9, mod 37, paired 9+37, shuffled labels. `angle_bin` runs as **CONTROL**, never evidence. `lock_claimed` is true only if a geometric residual stays small across mod 9 and mod 37 under `step_index`. On this orbit it does not. At 600 steps: `step_index` exNMI −0.022 and −0.082, paired/shuffled on the floor, `angle_bin` exNMI 0.719 / residual_R 0.175 tagged CONTROL, `lock_claimed=False` on every probe. Do not add moduli. `residual_R` is always a finite number (uninformative stamps get the uniform-circle null π/2, not infinity).

## CLI

```bash
python -m reveal header --steps N --seed S
python -m reveal names --steps N
python -m reveal all
```

`--full` uses 96 lattice sites. Tiny `N` is the test default. Load on the opening needs thousands of steps, for example:

```bash
python -m reveal header --full --steps 5000 --seed 0
python -m reveal names --steps 600
```

Writes `outputs/header.csv`, `outputs/header_trace.csv`, `outputs/names.csv`, `outputs/header_burst_rms.png`, `outputs/header_mean_theta.png`, `outputs/header_phi_b.png` (raw `α`, not a `φ_b` lock claim), `outputs/names_exnmi.png`.

On a long header run, read four numbers, not the smoke-plot y-scale: `burst_count`, late `rms_dTheta` (after 20% burn-in), `mean_Theta` / `mean_Theta_late`, and `first_mean_pi` / `first_mean_tcrit` / `first_burst_step`. Empty crossing fields mean the series never reached that threshold.

## Tests

```bash
pytest
```
