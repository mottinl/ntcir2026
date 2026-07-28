#!/bin/bash
# GPU0+GPU1 half of the split-by-advisor parallelization (see
# run_test_gpu2_paddleocr.sh for the other half): task2 ChartGemma ONLY.
# task1 dropped from this run entirely (user decision: finish task2 for
# both advisors as fast as possible, no time budget for task1 right now).
# PaddleOCR's task2 moved to GPU2 to run concurrently once it's free --
# partitioned by CONDITION, no file overlap with the GPU2 script.
#
# Resumable exactly like run_test_full.sh: re-running from scratch skips
# whatever's already done in predictions_task2_test_chartgemma.json.

set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=/data/models
export CUDA_VISIBLE_DEVICES=0,1
VENV_PY="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python"
MODEL=Qwen/Qwen3-VL-32B-Instruct

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== 1/1: task2 test, chartgemma (GPU0+1) ==="
$VENV_PY predict_task2.py --split test --model $MODEL --load-in-4bit \
    --advisor-cache advisor_cache_chartgemma.jsonl \
    --output predictions_task2_test_chartgemma.json > run_task2_test_chartgemma.log 2>&1
log "task2 chartgemma done (exit $?)"

log "=== GPU0+1 (chartgemma, task2 only) DONE ==="
