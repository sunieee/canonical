#!/usr/bin/env bash
set -euo pipefail

# Hyper-parameter tuning launcher for canonical/aggregation.py
# Usage:
#   bash canonical/tune_aggregation.sh
# Optional env vars:
#   DATASET=codex-m MODEL=LinearAggregator DEVICE=cuda MAX_PROCESSES=$(nproc)

DATASET="${DATASET:-codex-m}"
MODEL="${MODEL:-LinearAggregator}"
DEVICE="${DEVICE:-cuda}"
MAX_PROCESSES="${MAX_PROCESSES:-$(nproc)}"
MAX_WORKER_DATALOADER="${MAX_WORKER_DATALOADER:-$((MAX_PROCESSES>1?MAX_PROCESSES-1:1))}"

# Search space (edit as needed)
# POS_WEIGHTS=(5 15 30 100)
# LRS=(0.001 0.003 0.01)
POS_WEIGHTS=(5 15 30 100 400)
LRS=(0.001 0.01)
EPOCHS=(10 20)
SIGN_CONSTRAINTS=("" "--no_sign_constraint")

for pos_weight in "${POS_WEIGHTS[@]}"; do
  for lr in "${LRS[@]}"; do
    for max_epoch in "${EPOCHS[@]}"; do
      for sign_flag in "${SIGN_CONSTRAINTS[@]}"; do
        echo "[RUN] dataset=${DATASET} model=${MODEL} pos_weight=${pos_weight} lr=${lr} max_epoch=${max_epoch} sign='${sign_flag}'"

        python canonical/aggregation.py \
          --dataset "${DATASET}" \
          --model "${MODEL}" \
          --device "${DEVICE}" \
          --max_processes "${MAX_PROCESSES}" \
          --max_worker_dataloader "${MAX_WORKER_DATALOADER}" \
          --pos_weight "${pos_weight}" \
          --lr "${lr}" \
          --max_epoch "${max_epoch}" \
          ${sign_flag}
      done
    done
  done
done
