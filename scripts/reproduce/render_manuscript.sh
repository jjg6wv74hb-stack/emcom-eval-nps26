#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
manuscript_dir="$PWD/paper/neurips2026_comm_vecstraight"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$manuscript_dir/_output/mplconfig}"
mkdir -p "$MPLCONFIGDIR"
cd "$manuscript_dir"
quarto render main.qmd --to pdf --execute-daemon false
