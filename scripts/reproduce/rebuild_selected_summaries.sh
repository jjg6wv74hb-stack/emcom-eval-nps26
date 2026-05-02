#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
cat <<'MSG'
Rebuild one manuscript summary family at a time with the scripts in src/analysis/.
Examples:
  python -m src.analysis.summarize_phase3_base_gap_from_suite --help
  python -m src.analysis.summarize_phase3_sender_encoding_distribution --help
  python -m src.analysis.summarize_phase3_cross_seed_transfer --help

The checked-in artifacts under artifacts/paper/ are the canonical saved outputs used by the manuscript.
MSG
