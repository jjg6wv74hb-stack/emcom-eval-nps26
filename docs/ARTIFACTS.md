# Artifact Map

Saved outputs used by the manuscript are under `artifacts/paper/`. The
manuscript reads from these paths at render time.

The file `artifacts/paper/manifest.json` records file counts, byte sizes, and checksums for large files.

## Canonical Directories

- `base_gap/` - base communication gap suite and report.
- `endpoint/` - frozen endpoint intervention suites.
- `sender_encoding/` - encoding distribution outputs and figures.
- `sender_causal/` - single-sender causal probes.
- `message_source/` - message-source controls and full training/evaluation folders.
- `factorial/` - communication-by-history factorial runs and endpoint reductions.
- `transfer/` - cross-seed transfer and alignment summaries.
- `crossover/` - endpoint cross-over matrix analysis and per-cell reports.
- `noise_sweep/` - private/public observation-noise sweep outputs.
- `role_allocation/` - compact summary artifacts and figures for the hidden-need role-allocation second paradigm.
- `provenance/` - controller, manifest, validation, and status outputs.

Some provenance fields use generic run labels. Those fields are not used by the
manuscript render.

## Artifact Zip Subset

The artifact zip copies the compact CSV, JSON, JSONL, PDF, PNG, and status-log
files consumed by the manuscript render. Large checkpoints, raw traces, raw
suite folders, and full logs are not included.

The exported subset uses generic run labels in saved path fields. If `main.qmd`
reads a new artifact, rerun the render check from the unpacked zip before
submission.
