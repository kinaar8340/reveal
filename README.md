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

**A — header ablation.** Same `Δω`, `κ`, `θ_crit`, seed, steps. `headed` applies `q ← q · g(α)` after the two-gyro update; `unheaded` does not. Same burst reconnection as the toe two-gyro demo. The job fails if the two runs are bit-identical while the gauge path is supposed to be on. `W_g` / `φ_b` lock columns stay false: this harness does not spin up the torch conduit.

**B — name-change null.** Fixed geometry `9/π`, method `step_index`. Stamps: mod 9, mod 37, paired 9+37, shuffled labels. `angle_bin` runs as **CONTROL**, never evidence. `lock_claimed` is true only if a geometric residual stays small across mod 9 and mod 37 under `step_index`. On this orbit it does not. Do not add moduli.

## CLI

```bash
python -m reveal header --steps N --seed S
python -m reveal names --steps N
python -m reveal all
```

`--full` lengthens the defaults. Tiny `N` is the test default.

Writes `outputs/header.csv`, `outputs/names.csv`, `outputs/header_burst_rms.png`, `outputs/header_phi_b.png` (pointer / `α` stability, not a `φ_b` lock claim), `outputs/names_exnmi.png`.

## Tests

```bash
pytest
```
