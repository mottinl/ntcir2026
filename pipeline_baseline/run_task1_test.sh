#!/usr/bin/env bash
# Formal test-split run for Subtask 1. No eval step -- the test split has no
# gold labels (see evaluation_pipeline/data/test/data/, no *_task1_release
# ground truth file there, only the release JSON with the claims/images).
# Model + --load-in-4bit are pinned here on purpose: predict_task1.py falls
# back to the 8B default model if --model is omitted, and to an unsplit bf16
# load (known to OOM across these 2x20GB GPUs) if --load-in-4bit is omitted.
set -e
cd "$(dirname "$0")"
export HF_HOME=/data/models
MODEL="Qwen/Qwen3-VL-32B-Instruct"

echo "=== task1 (test, 4bit) starting $(date) ==="
../.venv/bin/python predict_task1.py --split test --model "$MODEL" --load-in-4bit \
    --output predictions_task1_test.json
echo "=== task1 (test, 4bit) done $(date) ==="
