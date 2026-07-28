# pipeline_advisors — Qwen3-VL decider + specialized chart/table advisor

Same foundation as `../pipeline_baseline/` (same guardrailed Rec-3 prompts,
same Qwen3-VL-32B-Instruct 4-bit decider on GPU0+GPU1, same evaluation
harness), plus an optional preliminary step: a specialized model "reads" the
image first, and that reading is injected into the decider's prompt as
auxiliary context (explicitly framed as fallible, not as ground truth).

Hypothesis under test: a specialized chart/table model that "reads" the
image first can ground the 32B decider better than a raw image reading
alone.

This README is the project's log, kept up to date with each decision/pivot
— see "History" at the end for the full timeline.

## Architecture

Two separate passes (the two models are never in memory at the same time —
the 32B decider already occupies GPU0+GPU1 at ~10-12 GiB/GPU once loaded,
leaving no headroom for a second model):

1. **Extraction** (`extract_advisor_cache.py`): one advisor at a time reads
   every image in its scope (figures or tables) and writes a resumable
   JSONL cache (`advisor_cache_<name>.jsonl`).
2. **Decision** (`predict_task1.py` / `predict_task2.py`, copied from
   pipeline_baseline + `--advisor-cache`): identical to pipeline_baseline if
   the option is omitted (verified byte-identical control), otherwise
   injects the cached reading into the prompt (a block explicitly labelled
   "auxiliary, NOT ground truth").

task1 was dropped from active scope partway through (see History): the
pipeline now only runs for task2. task1's code (`predict_task1.py`, the
advisor block in `prompt.txt`) remains functional and verified
byte-identical, simply unused for now.

## Advisors used

| Model | Scope | Interface | Notes |
|---|---|---|---|
| `ahmed-masry/chartgemma` | figures | `PaliGemmaForConditionalGeneration` (`transformers`, shared venv) | instruction-tuned, free-form prompt |
| PaddleOCR / PP-Structure `TableRecognitionPipelineV2` | tables | `paddleocr` (**isolated** venv, GPU) | dedicated table structure + OCR, not a VLM |

### PaddleOCR/PP-Structure — isolated venv (required)

`paddlepaddle-gpu` installs its own pinned `nvidia-cu*` versions
(cublas/cudnn/...) that conflict with the shared venv's torch build —
**confirmed by a real incident**: installing it into the shared venv
downgraded `nvidia-cublas`/`nvidia-cudnn` out from under torch while a
decider run was in progress; fixed via `uv sync` before it affected any
downstream step. Hence a dedicated venv:

```bash
uv venv pipeline_advisors/.venv-paddleocr --python 3.13
uv pip install --python pipeline_advisors/.venv-paddleocr/bin/python \
  "paddlepaddle-gpu==3.3.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install --python pipeline_advisors/.venv-paddleocr/bin/python paddleocr "paddlex[ocr]" pillow
```

**CPU is not viable**: measured at 30s-200s+/image (some images never
finished within a 3+ minute timeout) due to a real paddlepaddle 3.3.x
regression on CPU oneDNN inference
([Paddle#77340](https://github.com/PaddlePaddle/Paddle/issues/77340)) —
worked around by pinning `paddlepaddle==3.2.0` (CPU, shared venv, for any
future CPU use). On GPU (a different execution path, no oneDNN/MKLDNN
involved), no issue at all: **~3.5s/image regardless of table complexity**,
including the cases that stalled on CPU. GPU is required for this advisor
at this corpus size (1005 images).

`advisor_paddleocr.py` uses `TableRecognitionPipelineV2(use_layout_detection=False, ...)`
since the input image is already a cropped table (not a full page to lay
out) — output is the table's HTML (`pred_html`), injected as-is into the
prompt (an LLM reads table HTML without issue).

### Table-LLaVA — replaced by PaddleOCR

`SpursgoZmy/table-llava-v1.5-7b` (code kept: `advisor_table_llava.py`,
`convert_table_llava.py`) was tested thoroughly then abandoned:
- **~7.5% (75/1005)** of extractions showed degenerate collapse (looping
  repetition of empty/nonsensical rows) on wide tables, consistent with its
  fixed 336×336 input resolution (well below Qwen3-VL's).
- Even on "clean"-looking outputs, manual verification against the real
  images (`val_tab_0026`): fabricated headers, entire columns dropped
  (`Heavy Rain`), row labels replaced with invented numbers — whereas
  PaddleOCR, on the same image, reproduces the exact structure (merged
  Rain/Heavy Rain headers, all columns) and the numeric values across
  several rows.
- Even on a pathological case common to both (`val_tab_0005`, an atypical
  heatmap-over-text table): PaddleOCR reproduces text fragments genuinely
  present in the image (in scrambled order), whereas Table-LLaVA had
  fabricated entirely hallucinated, unrelated content.

The code is kept on disk (not deleted, not used) in case it's worth
reconsidering — `convert_table_llava.py` documents in detail the 4 bugs
found and fixed while converting the checkpoint.

### Paths not taken / possible future work

- **DePlot** (`google/deplot`, `Pix2StructForConditionalGeneration`) was
  implemented and smoke-tested (5 images, plausible linearized-table output)
  but **dropped without a formal head-to-head comparison** against
  ChartGemma on dev — a deliberate choice to move faster rather than a real
  negative result. Worth reconsidering if ChartGemma's scores disappoint:
  the code was removed (`advisor_deplot.py` no longer exists) but followed
  the exact same pattern as the other advisors (`Advisor` class,
  `SCOPE="figure"`, fixed prompt `"Generate underlying data table of the
  figure below:"`), easy to rewrite if needed.
- **TinyChart-3B** ruled out at the planning stage (no native `transformers`
  support, requires the custom `tinychart` package from
  github.com/X-PLUG/mPLUG-DocOwl — integration risk judged too high for a
  first attempt).

## Usage

Extraction (pass A, once per advisor, resumable):

```bash
# figures -- shared venv
python extract_advisor_cache.py --advisor chartgemma --limit 5   # smoke test
python extract_advisor_cache.py --advisor chartgemma

# tables -- ISOLATED venv (see PaddleOCR section above)
.venv-paddleocr/bin/python extract_advisor_cache.py --advisor paddleocr
```

Full run on the official test split, one script per GPU/advisor, detached
and resumable (rerunning picks up exactly where it left off, via the
"done" set read back from `--output`). **Finished (stopped for good, see
History point 11)** — kept here as a record of how
`predictions_task2_test_{chartgemma,paddleocr}.json` were produced, and
directly reusable if GPUs become available again:

```bash
setsid nohup ./run_test_gpu01_chartgemma.sh > run_gpu01_chartgemma.log 2>&1 < /dev/null & disown
setsid nohup ./run_test_gpu2_paddleocr.sh    > run_gpu2_paddleocr.log    2>&1 < /dev/null & disown
```

- `run_test_gpu01_chartgemma.sh`: task2 test, ChartGemma, `CUDA_VISIBLE_DEVICES=0,1`.
- `run_test_gpu2_paddleocr.sh`: task2 test, PaddleOCR, `CUDA_VISIBLE_DEVICES=2`
  (single-GPU `device_map="auto"`, see `qwen_agent.py::_Bnb4BitQwenAgent` —
  automatically switches between 1- and 2-GPU placement based on
  `torch.cuda.device_count()`, no code change required). GPU2 was shared with
  other ad hoc tasks on this host, so this script can be stopped and
  restarted without loss: `predict_task2.py` saves every 10 examples
  (`--save-every`) and resumes from `--output` on restart.

Manual prediction (debug/smoke-test), including restricted to one
`evi_type` to compare an advisor on a targeted subset:

```bash
python predict_task2.py --split test --model Qwen/Qwen3-VL-32B-Instruct --load-in-4bit \
  --advisor-cache advisor_cache_chartgemma.jsonl --output predictions_task2_test_chartgemma.json
python predict_task2.py --split dev --evi-type figure --limit 10 --advisor-cache advisor_cache_chartgemma.jsonl
```

### Final merge by `evi_type` (`merge_specialized_by_evi_type.py`)

Every task2 sample is entirely table OR figure (never mixed — verified
across all 436 samples of the official test split: 256 table / 180 figure).
The final deliverable routes each sample to the relevant advisor:

```bash
python merge_specialized_by_evi_type.py --output predictions_task2_test.json
```

For each sample: figure -> ChartGemma prediction if available, table ->
PaddleOCR prediction if available, otherwise (run stopped before reaching
that sample — see History, hard stop at 370/436 per condition) fall back to
`pipeline_baseline/predictions_task2_test.json` (a complete 436/436 result
already computed by pipeline_baseline, chosen as the fallback because it
fixes known parsing/fallback errors present on ~120 samples of an earlier
run — better quality even though its prompts weren't verified
byte-identical to pipeline_advisors' own). No id is ever missing, no value
is ever fabricated.

`merge_with_baseline_fallback.py` remains on disk as a generic utility
(one advisor at a time, task1 or task2, configurable per-id fallback) but
wasn't used to produce the final deliverable — superseded here by the
per-`evi_type` logic above.

## Results

- `predictions_task2_test.json` — **final deliverable** (436/436): every
  sample uses its specialized advisor (ChartGemma for figures, PaddleOCR for
  tables), falling back to pipeline_baseline's dual-prompt result for the
  rest. Composition as of the final stop (see History): 155 ChartGemma + 215
  PaddleOCR + 66 fallback.
- `predictions_task2_test_chartgemma.json` / `predictions_task2_test_paddleocr.json`
  — raw output of each run, each stopped at 370/436 (see History) — kept as
  intermediate reference files.
- `predictions_task2_test_no_advisor.json` — no advisor, copied as-is from
  pipeline_baseline's single-joint-prompt run (prompts verified
  byte-identical to pipeline_advisors'). Serves as a zero-shot comparison
  reference, not used as the fallback in the final deliverable (see above).

No local score: the test split isn't labelled, scoring is the task
organizers' responsibility. The labelled `dev` split remains available for
a real score if needed (`run_eval.py`), including restricted by `--evi-type`.

## History (chronology of decisions)

1. Scaffolded from `pipeline_baseline/` (same prompts/decider/harness),
   two-pass architecture so the advisor and the 32B decider are never
   loaded together.
2. Figure advisors considered: DePlot + ChartGemma. Table advisor
   considered: Table-LLaVA (TinyChart-3B ruled out upfront, see "Paths not
   taken").
3. DePlot smoke-tested then dropped without a formal comparison (to move
   faster) — ChartGemma kept alone for figures.
4. Table-LLaVA converted (native HF checkpoint, 4 real bugs found and
   documented in `convert_table_llava.py`), fully extracted, then evaluated
   qualitatively on real images: ~7.5% degenerate outputs and structural
   hallucinations even on "clean" outputs -> replaced by PaddleOCR/PP-Structure
   (clearly superior on the same images, see dedicated section above).
5. PaddleOCR integrated via an isolated venv after a near-incident (installing
   `paddlepaddle-gpu` in the shared venv downgraded torch's CUDA libs while a
   run was in progress — caught and fixed immediately via `uv sync`, before
   any damage).
6. `task1` dropped from active scope: time constraint, priority given to
   `task2` only for this run.
7. Parallelized by condition (no intra-task sharding): ChartGemma on
   GPU0+GPU1, PaddleOCR on GPU2 (once that 80GB GPU became free) — disjoint
   output files, no write-race risk.
8. GPU2 temporarily reclaimed for an external priority task: the PaddleOCR
   run was stopped cleanly right after a checkpoint save (no loss), then
   resumed later exactly where it left off.
9. Both task2 runs (ChartGemma, PaddleOCR) were then deliberately stopped
   partway through (250/436 and 200/436) to produce a single deliverable as
   soon as possible: `merge_specialized_by_evi_type.py`, which routes each
   sample to its specialized advisor by `evi_type` and fills the rest with
   pipeline_baseline's already-complete dual-prompt result. Result saved as
   `_old` as a precaution, then both runs resumed in the background to keep
   reducing the fallback share.
10. Cleanup (2026-07-20): removed dead/obsolete files (a sequential
    full-run script that still referenced task1 after its removal from
    scope; an extraction script for an advisor removed from the registry;
    one-off extraction logs already summarized here; intermediate merge
    files superseded by the per-`evi_type` approach; a task1 "no advisor"
    copy, task1 being out of scope). `.gitignore` widened to cover isolated
    venvs (`.venv-*`).
11. Both runs resumed in the background after point 9, until a hard stop on
    2026-07-20 around 19:35: the host's GPUs were reallocated to another
    server for an external need (not a crash) — `nvidia-smi` failing to
    reach the driver after the following reboot was the observed trace of
    this. Last clean save (every 10 examples): 370/436 for both ChartGemma
    and PaddleOCR — no loss, valid files.
12. Continuing on CPU (56 cores, 62 GiB RAM) was considered to finish the
    remaining 66 examples per condition without a GPU, but **judged not
    viable after a real test**: a 1-example smoke test (`--load-in-4bit` on
    CPU) was OOM-killed (signal 9) after ~2 min, still mid weight-loading,
    at 63 GiB resident on a 62 GiB box — bitsandbytes' CPU path for this 32B
    model simply doesn't fit in this machine's RAM (not just "slow").
    Decision: don't requantize to GGUF for the 132 remaining examples,
    fill with the dual-prompt fallback instead (point 13).
13. Final deliverable produced via `merge_specialized_by_evi_type.py` ->
    `predictions_task2_test.json` (436/436: 155 ChartGemma + 215 PaddleOCR +
    66 pipeline_baseline dual-prompt fallback, see "Results"). Final cleanup:
    removed the superseded specialized/`_old` prediction files, intermediate
    `*_merged.json` merge files, per-run logs, a finished extraction script,
    and `__pycache__/`.
