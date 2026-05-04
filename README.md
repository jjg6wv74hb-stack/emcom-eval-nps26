# EPGG Emergent Communication Evaluation

This repository contains code, manuscript sources, and saved result artifacts for the paper *What Does Emergent Communication Actually Communicate? An Evaluation Framework Separating Channel Structure from Learned Content*.

## Quick Start

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements_locked.txt
./scripts/reproduce/render_manuscript.sh
```

See `docs/COMMANDS.md` for runnable commands, `docs/REPRODUCIBILITY.md`
for installation and scope, and `docs/ARTIFACTS.md` for the artifact map.

## Review Artifact

The review artifact contains the source code, tests, documentation, manuscript
files, role-allocation code/configs, and the summary artifacts used by the
manuscript render. It excludes large checkpoints, raw traces, local caches, and
generated outputs.

## Layout

- `src/`: environment, training, evaluation, and analysis code.
- `tests/`: unit and regression tests for scientific contracts and analysis helpers.
- `paper/neurips2026_comm_vecstraight/`: Quarto manuscript source and figure assets.
- `artifacts/paper/`: saved training/evaluation outputs used by the manuscript. Large files are intended for Git LFS.
- `scripts/reproduce/`: rendering and check scripts.
- `docs/`: reproducibility and artifact documentation.

## Manuscript Data

The manuscript reads its reported numbers from the saved outputs in
`artifacts/paper/`.
