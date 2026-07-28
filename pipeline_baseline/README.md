# pipeline_baseline — Qwen3-VL single-stage baseline

Single-stage baseline: the claim's evidence image (table or figure) is given
directly to a Qwen3-VL checkpoint, along with the claim, caption and paper
context. No separate OCR/chart-extraction step — the VLM reads the image
itself. This is the simpler of the two submitted approaches; see
`../pipeline_advisors/` for the variant that adds a specialized "advisor"
model as an auxiliary reading.

Uses the shared repo-root `.venv` (transformers, torch, vllm, bitsandbytes
already there). No extra install needed to run.

## Files

- `qwen_agent.py` — loads the model once, exposes `generate(images, text)`.
  Dispatches to one of three backends based on the checkpoint/flags (see
  "Model & backends" below); callers don't need to know which one loaded.
- `data_utils.py` — resolves paths against `evaluation_pipeline/data/<split>/`
- `parsing.py` — extracts the final `Supported`/`Refuted` or
  `evidence_id_1`/`evidence_id_2` label from the model's response
- `predict_task1.py` — claim label prediction (Subtask 1), prompt template
  pulled from `prompt.txt` (`REC3_USER_PROMPT_TEMPLATE`)
- `predict_task2.py` — evidence selection (Subtask 2), prompt templates
  pulled from `prompt_task2.txt` (`TASK2_GENERIC_PROMPT_TEMPLATE` for tables,
  `TASK2_FIGURE_PROMPT_TEMPLATE` for figures, routed by `evi_type`)
- `predict_task2_decomposed.py` — alternate task2 strategy: classify each of
  the two candidate images independently with task1's prompt, and only fall
  back to the joint two-image prompt (`predict_task2.py`'s approach) on a
  tie. Rationale: task2's two candidates are always the genuine evidence and
  a perturbed distractor for the same claim, so task1's Supported/Refuted
  call on each image in isolation is often decisive on its own.
  `test_task2_decomposed.py` covers the branching logic with a scripted
  stub agent (CPU only, no model loaded) — it was validated this way but
  never run end-to-end on GPU, so it wasn't used for the submitted run.
- `merge_shards.py` — merge `--num-shards`-sharded prediction files back into one
- `run_task1_test.sh` / `run_task2_test.sh` — formal test-split runs (see
  "Running" below)
- `run_dev_eval.sh` — dev-split predictions + scoring against gold labels,
  reproduces `eval_results_dev.txt`

## Model & backends

`qwen_agent.QwenVLAgent` picks a backend at construction time:

| Backend | Used when | Checkpoint example |
|---|---|---|
| plain transformers | default (no `--load-in-4bit`, non-AWQ checkpoint) | `Qwen/Qwen3-VL-8B-Instruct` |
| vLLM (AWQ, Marlin kernel, single GPU + CPU offload) | checkpoint's config reports `quant_method: awq` | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` |
| transformers + bitsandbytes 4-bit, explicit 2-GPU split | `--load-in-4bit` passed | `Qwen/Qwen3-VL-32B-Instruct` |

**The submitted runs use the third path**: `Qwen/Qwen3-VL-32B-Instruct`
(non-quantized checkpoint) loaded on-the-fly in 4-bit via bitsandbytes,
decoder layers split evenly across both GPUs (vision tower stays bf16 —
quantizing it produces broken `Linear4bit` layers at generate() time, see
`qwen_agent.py` docstring):

```bash
cd pipeline_baseline
export HF_HOME=/data/models   # model is already cached locally; avoids depending on network access
python predict_task1.py --split dev --model Qwen/Qwen3-VL-32B-Instruct \
    --load-in-4bit --output predictions_task1_dev.json
python predict_task2.py --split dev --model Qwen/Qwen3-VL-32B-Instruct \
    --load-in-4bit --output predictions_task2_dev.json
```

An earlier AWQ-quantized checkpoint (`QuantTrio/Qwen3-VL-32B-Instruct-AWQ`
via vLLM) was tried first and dropped: this host's GPUs are virtualized
(GRID vGPU) slices that reject vLLM's NCCL-based multi-GPU init, so the AWQ
path was restricted to a single GPU — tight enough to OOM on this dataset's
largest images even with `cpu_offload_gb`, and it scored far below the
bitsandbytes 4-bit path on dev (accuracy ~48%, pair_accuracy ~2.6% on task2).

Decoding uses Qwen's recommended sampling for Instruct checkpoints
(temperature=0.7, top_p=0.8, top_k=20). Pass `--greedy` for greedy decoding
instead. A fixed `--seed` (default 0) makes a given run reproducible.

## Running

Smoke test on a handful of dev examples first (dev has gold labels, so you
can sanity-check the outputs by eye or run `evaluation_pipeline/eval/run_eval.py`
against them):

```bash
cd pipeline_baseline
source ../.venv/bin/activate
python predict_task1.py --split dev --model Qwen/Qwen3-VL-32B-Instruct --load-in-4bit --limit 5
python predict_task2.py --split dev --model Qwen/Qwen3-VL-32B-Instruct --load-in-4bit --limit 5
```

Full formal run (test split, no gold labels — see `run_task1_test.sh` /
`run_task2_test.sh`; safe to Ctrl-C and re-run, it resumes from what's
already in the output file):

```bash
./run_task1_test.sh
./run_task2_test.sh
```

For a faster task2 test run, shard across GPUs and merge the parts
afterward (`--num-shards N --shard-index i` per process, then
`merge_shards.py`) — see the comment at the bottom of `run_task2_test.sh`.

## Evaluating against dev gold labels

```bash
./run_dev_eval.sh
```

or manually:

```bash
cd ../evaluation_pipeline/eval
python run_eval.py --task task1 \
  --ground_truth_task1 ../data/dev/data/dev_task1_release.json \
  --pred_task1 ../../pipeline_baseline/predictions_task1_dev.json

python run_eval.py --task task2 \
  --ground_truth_task2 ../data/dev/data/dev_task2_release.json \
  --pred_task2 ../../pipeline_baseline/predictions_task2_dev.json
```

Dev scores (`Qwen/Qwen3-VL-32B-Instruct`, 4-bit, see `eval_results_dev.txt`):
task1 precision 80.78 / recall 80.75 / macro_f1 80.45 / accuracy 80.46 /
pair_accuracy 65.34; task2 accuracy 67.61.

An error-analysis pass on the task1 dev predictions found an asymmetric
confusion matrix (`[[302,50],[96,299]]`): the model is biased toward
predicting "Refuted" (96 gold-Supported claims predicted Refuted, vs. only
50 in the other direction), fairly evenly split across evidence types and
domains — a possible target for future prompt tuning.

## Notes

- If the model's response doesn't contain a clearly parseable final label
  (should be rare — the prompt asks for an exact `Final answer: ...` line),
  the script logs a warning and falls back to a fixed default (`Refuted` for
  task1, `evidence_id_1` for task2) so the run never crashes or produces an
  invalid submission. The final log line reports how many fallbacks occurred
  — if that number isn't ~0, look at the warnings before trusting the scores.
  The submitted `predictions_task2_test.json` reflects a rerun (at a larger
  `--max-new-tokens`) of the samples that first fell back due to truncation,
  with any still-unresolved cases patched in from an earlier, non-routed
  prompt version rather than left on the arbitrary default.
- `--max-new-tokens` defaults differ per script: 10240 for task1 (long
  guardrailed reasoning prompt), 2048 for task2. Qwen's own recommendation for
  VL tasks goes up to 40960; raise further if you see truncated/no
  `Final answer:` lines in the warnings.
- No batching yet — one example at a time. Fine for correctness-first testing;
  worth revisiting if a full test-set run turns out too slow.
- `--load-in-4bit` splits decoder layers across exactly 2 devices when 2+
  GPUs are visible (see `_Bnb4BitQwenAgent._build_device_map` in
  `qwen_agent.py`); with `CUDA_VISIBLE_DEVICES` masked down to a single GPU
  it falls back to `device_map="auto"` instead, which is what all the
  single-80GB-GPU test runs above actually used — that path is validated at
  this checkpoint size, not just the 2-GPU one.
