#!/usr/bin/env bash
#
# Reproduce Table 3 (ablation study on TimesNet × 3 datasets, paper §4.3).
#
# Four columns:
#   PILOT          — no ablation
#   w/o Scorer     — --ablation_variant wo_learning_oracle
#   w/o History    — --ablation_variant wo_historical
#   w/o Stacking   — --ablation_variant n1
#
# Usage:
#   bash scripts/reproduce_table3.sh [OUTPUT_ROOT] [SEED]

set -euo pipefail

OUTPUT_ROOT="${1:-./results_table3}"
SEED="${2:-42}"
DATA_ROOT="${DATA_ROOT:-./data}"
BACKBONE="timesnet"
LABELS="mean_deg,cum_deg,rel_deg,binary_utility,ordinal_utility"

mkdir -p "${OUTPUT_ROOT}"

run_variant() {
    local variant_tag="$1"
    local variant_flag="$2"
    for DATASET in ETTh2 ETTm1 Traffic; do
        OUT="${OUTPUT_ROOT}/${variant_tag}/${BACKBONE}_${DATASET}"
        echo "==[$(date +%H:%M:%S)] ${variant_tag} ${BACKBONE}/${DATASET} seed=${SEED}"
        python scripts/run_method.py \
            --backbone "${BACKBONE}" \
            --datasets "${DATASET}" \
            --seed "${SEED}" \
            --label_types "${LABELS}" \
            --benchmarks_dir "${DATA_ROOT}" \
            --output_dir "${OUT}" \
            ${variant_flag}
    done
}

run_variant pilot           ""
run_variant wo_scorer       "--ablation_variant wo_learning_oracle"
run_variant wo_history      "--ablation_variant wo_historical"
run_variant wo_stacking     "--ablation_variant n1"

echo "==[$(date +%H:%M:%S)] Table 3 reproduction done."
