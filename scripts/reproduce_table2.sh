#!/usr/bin/env bash
#
# Reproduce Table 2 (trigger-quality metrics, paper §4.4).
#
# Reads the predictions.csv / retrains.csv files produced by
# reproduce_table1.sh and writes a single trigger_quality.csv with per-cell
# precision / trig_need / ep_hit / det_lag_hours for every method.
#
# Usage:
#   bash scripts/reproduce_table2.sh [TABLE1_OUTPUT_ROOT] [SEED]

set -euo pipefail

TABLE1_ROOT="${1:-./results_table1}"
SEED="${2:-42}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_CSV="${TABLE1_ROOT}/trigger_quality.csv"

python scripts/compute_trigger_quality.py \
    --output_root "${TABLE1_ROOT}" \
    --benchmarks_dir "${DATA_ROOT}" \
    --seed "${SEED}" \
    --out "${OUT_CSV}"
