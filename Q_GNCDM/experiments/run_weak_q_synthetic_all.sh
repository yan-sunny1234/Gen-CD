#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/synthetic_weak_q/seed_2026}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-2026}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for noise in 0.05 0.10 0.20 0.30 0.40; do
  for mode in fixed-noisy-q weak-q oracle-q free-q; do
    "${PYTHON_BIN}" experiments/run_weak_q_synthetic.py \
      --data_dir "${DATA_DIR}" \
      --noise_rate "${noise}" \
      --mode "${mode}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}"
  done
done

