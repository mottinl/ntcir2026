#!/bin/bash
# GPU2 half of the split-by-advisor parallelization (see
# run_test_gpu01_chartgemma.sh for the other half): task2 PaddleOCR ONLY,
# running as a SINGLE-GPU decider instance on GPU2 alone. task1 dropped from
# this run entirely (user decision: finish task2 for both advisors as fast
# as possible, no time budget for task1 right now).
#
# This is plain transformers + bitsandbytes 4-bit (qwen_agent.py's
# _Bnb4BitQwenAgent): with only 1 GPU visible (CUDA_VISIBLE_DEVICES=2),
# torch.cuda.device_count()==1 so it automatically falls back to
# device_map="auto" (single-GPU placement) instead of the explicit 2-GPU
# layer split -- no code change needed, this is already handled by the
# existing agent code. A single 80GB GPU comfortably fits the whole
# ~20-24GiB footprint with room to spare, and may even be FASTER per example
# than the GPU0+1 split (no inter-GPU tensor transfers between decoder
# layers every forward pass).
#
# IMPORTANT: only launch this once GPU2 is confirmed free (it's shared with
# a third-party task on this host) -- check `nvidia-smi` first. Uses the
# SHARED repo-root .venv (same as the GPU0+1 script) -- the isolated
# .venv-paddleocr/ was only ever needed for the PaddleOCR *extraction* step
# (already finished, cache is a plain JSONL file, readable by anyone), not
# for the decider here.
#
# Partitioned by CONDITION (not sharded within a task) specifically so this
# never writes to the same output file as run_test_gpu01_chartgemma.sh --
# no race/collision possible between the two concurrent scripts.
#
# Resumable exactly like run_test_full.sh: re-running from scratch skips
# whatever's already done in predictions_task2_test_paddleocr.json.

set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=/data/models
export CUDA_VISIBLE_DEVICES=2
VENV_PY="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python"
MODEL=Qwen/Qwen3-VL-32B-Instruct

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== 1/1: task2 test, paddleocr (GPU2) ==="
$VENV_PY predict_task2.py --split test --model $MODEL --load-in-4bit \
    --advisor-cache advisor_cache_paddleocr.jsonl \
    --output predictions_task2_test_paddleocr.json > run_task2_test_paddleocr.log 2>&1
log "task2 paddleocr done (exit $?)"

log "=== GPU2 (paddleocr, task2 only) DONE ==="
