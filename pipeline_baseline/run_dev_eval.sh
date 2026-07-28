#!/usr/bin/env bash
# Reproduces predictions_task{1,2}_dev.json and eval_results_dev.txt: runs
# both subtasks on the labelled dev split, then scores them against the
# official gold labels. Useful to sanity-check the setup (model weights,
# GPU split, prompts) before committing to a full, unscored test-split run.
set -e
cd "$(dirname "$0")"
export HF_HOME=/data/models
MODEL="Qwen/Qwen3-VL-32B-Instruct"

echo "=== task1 (dev, 4bit) starting $(date) ==="
../.venv/bin/python predict_task1.py --split dev --model "$MODEL" --load-in-4bit \
    --output predictions_task1_dev.json
echo "=== task1 (dev, 4bit) done $(date) ==="

echo "=== task2 (dev, 4bit) starting $(date) ==="
../.venv/bin/python predict_task2.py --split dev --model "$MODEL" --load-in-4bit \
    --output predictions_task2_dev.json
echo "=== task2 (dev, 4bit) done $(date) ==="

echo "=== evaluation starting $(date) ==="
cd ../evaluation_pipeline/eval
../../.venv/bin/python run_eval.py --task both \
    --ground_truth_task1 ../data/dev/data/dev_task1_release.json \
    --pred_task1 ../../pipeline_baseline/predictions_task1_dev.json \
    --ground_truth_task2 ../data/dev/data/dev_task2_release.json \
    --pred_task2 ../../pipeline_baseline/predictions_task2_dev.json \
    | tee ../../pipeline_baseline/eval_results_dev.txt
echo "=== evaluation done $(date) ==="
