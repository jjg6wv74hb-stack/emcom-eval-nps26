# Reproducibility

## Environment

Use Python 3.10. The locked requirements are in `requirements_locked.txt`; the
shorter direct dependency list is in `requirements.txt`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements_locked.txt
```

Quarto and a LaTeX distribution are required to render the manuscript PDF.

## Review Artifact

The review artifact contains source, tests, docs, manuscript sources,
role-allocation code/configs, and the summary artifacts read by
`paper/neurips2026_comm_vecstraight/main.qmd`. It does not include large
checkpoints, raw traces, full training logs, local caches, or generated Quarto
outputs.

## Render the Manuscript

From the repository root:

```bash
./scripts/reproduce/render_manuscript.sh
```

From an unpacked artifact zip:

```bash
cd paper/neurips2026_comm_vecstraight
MPLCONFIGDIR="$PWD/_output/mplconfig" quarto render main.qmd --to pdf --execute-daemon false
```

The manuscript reads quantitative results from `artifacts/paper/...` using
repo-relative paths. Rendered output is written under
`paper/neurips2026_comm_vecstraight/_output/`, which is ignored by Git and not
included in the artifact zip.

## Quick Verification

Suggested local checks:

```bash
python3 -m py_compile $(find src tests scripts -name '*.py')
python3 -m pytest tests/test_wrapper.py tests/test_role_allocation_env.py tests/test_role_allocation_vectorized_training.py -q
./scripts/reproduce/render_manuscript.sh
```

The full test suite can be run with:

```bash
python3 -m pytest -q
```

Some subprocess/vectorized-backend tests can be sensitive to OpenMP shared
memory limits on macOS or constrained machines. On macOS those tests are
skipped by default; set `EPGG_RUN_SUBPROC_TESTS=1` to force them. The serial
tests and manuscript render are the shortest checks.

For a shorter command sheet, see `docs/COMMANDS.md`.

## What Is Reproducible From This Artifact

The artifact zip supports manuscript rendering from saved summary artifacts and
running the included unit/regression tests. It does not include the full raw
trace/checkpoint tree needed to rerun every analysis from scratch; that tree is
about 15 GB.

Full retraining is computationally expensive. The EPGG training families use 15
seeds per condition and checkpoints up to 150k episodes. The manuscript appendix
reports batch-level wall-clock estimates from saved status logs. Rebuild one
family at a time with the relevant scripts under `src/experiments_pgg_v0/` and
`src/analysis/`, then compare the generated summaries with the checked-in CSVs.
`docs/COMMANDS.md` gives smoke-run commands and full-training command templates.

The role-allocation probe code and base config are included under
`src/environments/role_allocation/`, `src/experiments_role_allocation/`, and
`configs/role_allocation/`. The manuscript render uses compact role-allocation
summary CSVs and figures under `artifacts/paper/role_allocation/...`.
