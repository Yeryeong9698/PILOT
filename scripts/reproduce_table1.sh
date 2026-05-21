#!/usr/bin/env bash
#
# Reproduce Table 1 (main MSE results, 9 cells = 3 backbones × 3 datasets).
#
# Each invocation runs the five PILOT single-label variants plus voting,
# together with the six baselines, on one (backbone, dataset) cell.
# Outputs land under
# OUTPUT_ROOT/<cell>_<label>/results/<dataset>/<method>/seed_<SEED>/.
#
# Usage:
#   bash scripts/reproduce_table1.sh [OUTPUT_ROOT] [SEED]
#
# Defaults: OUTPUT_ROOT=./results_table1, SEED=42.
#
# The paper reports the mean over three random seeds.  Re-run with
# different SEED values (e.g. 42, 43, 44) and average the per-cell MSE
# values from results/<dataset>/<method>/seed_*/metrics.json.

set -euo pipefail

OUTPUT_ROOT="${1:-./results_table1}"
SEED="${2:-42}"
DATA_ROOT="${DATA_ROOT:-./data}"

LABELS="mean_deg,cum_deg,rel_deg,binary_utility,ordinal_utility"

mkdir -p "${OUTPUT_ROOT}"

for BACKBONE in dlinear itransformer timesnet; do
    for DATASET in ETTh2 ETTm1 Traffic; do
        CELL="${BACKBONE}_${DATASET}"
        OUT="${OUTPUT_ROOT}/${CELL}"
        echo "==[$(date +%H:%M:%S)] ${CELL} seed=${SEED}"
        python scripts/run_method.py \
            --backbone "${BACKBONE}" \
            --datasets "${DATASET}" \
            --seed "${SEED}" \
            --label_types "${LABELS}" \
            --benchmarks_dir "${DATA_ROOT}" \
            --output_dir "${OUT}"
    done
done

echo "==[$(date +%H:%M:%S)] Table 1 reproduction done."
echo "Per-cell MSE: ${OUTPUT_ROOT}/<cell>_<label>/results/<dataset>/<method>/seed_${SEED}/metrics.json"
