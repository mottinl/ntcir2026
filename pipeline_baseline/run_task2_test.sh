#!/usr/bin/env bash
# Formal test-split run for Subtask 2. No eval step -- the test split has no
# gold labels (see evaluation_pipeline/data/test/data/, no *_task2_release
# ground truth file there, only the release JSON with the claim/evidence pairs).
# Model + --load-in-4bit are pinned here on purpose: predict_task2.py falls
# back to the 8B default model if --model is omitted, and to an unsplit bf16
# load (known to OOM across these 2x20GB GPUs) if --load-in-4bit is omitted.
set -e
cd "$(dirname "$0")"
export HF_HOME=/data/models
MODEL="Qwen/Qwen3-VL-32B-Instruct"

echo "=== task2 (test, 4bit) starting $(date) ==="
../.venv/bin/python predict_task2.py --split test --model "$MODEL" --load-in-4bit \
    --output predictions_task2_test.json
echo "=== task2 (test, 4bit) done $(date) ==="
# For a faster run, shard across GPUs and merge the parts:
#   predict_task2.py ... --num-shards 3 --shard-index {0,1,2}  (one process per GPU)
#   merge_shards.py predictions_task2_test_shard*of3.json -o predictions_task2_test.json
