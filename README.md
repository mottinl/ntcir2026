# NTCIR-19 SciClaimEval — scientific claim verification with Qwen3-VL

Our submission to [SciClaimEval](https://sciclaimeval.github.io/) (NTCIR-19),
a shared task on verifying scientific claims against multi-modal evidence
(tables and figures from research papers). This repo holds the scripts,
prompts, and prediction files behind the runs we submitted, plus one related
exploratory pipeline.

## The task

Given a textual claim and its evidence (a table or figure image from a
paper, with its caption and surrounding context), SciClaimEval has two
subtasks:

- **Subtask 1 — Claim Label Prediction**: is the claim `Supported` or
  `Refuted` by the evidence?
- **Subtask 2 — Claim-Evidence Prediction**: given a claim and two
  candidate evidence images (the genuine one and a perturbed distractor),
  which one (`evidence_id_1` or `evidence_id_2`) actually supports the claim?

The corpus spans three domains: `nlp`/`ml` (arXiv/ACL papers) and `peerj`
(biomedical, from PubMed Central). Systems are evaluated on a labelled `dev`
split (for local validation) and an unlabelled `test` split (the official
formal run, scored by the task organizers). Subtask 1's primary metric is
**pair accuracy** (both the Supported and Refuted version of a claim must be
predicted correctly); Subtask 2's is plain accuracy.

## Repository layout

```
├── pipeline_baseline/    Single-stage baseline: Qwen3-VL-32B reads the
│                         evidence image directly. Produced the submitted
│                         Subtask 1 run and part of the Subtask 2 fallback.
├── pipeline_advisors/    Same decider, plus an optional specialized
│                         "advisor" model (ChartGemma for figures, PaddleOCR
│                         for tables) that reads the image first and feeds
│                         its reading into the decider's prompt as an
│                         auxiliary hint. Produced the submitted Subtask 2 run.
├── pipeline_sibils/      Exploratory, separate from the above: builds a
│                         synthetic training set from SIBiLS/PMC literature
│                         and fine-tunes Qwen3-VL-8B with QLoRA. Did not
│                         produce the submitted runs -- kept as a documented
│                         experiment (see its README for the outcome).
├── evaluation_pipeline/  Dataset (via symlinks) + the task organizers'
│                         vendored scoring harness (run_eval.py).
└── analysis/             Standalone dataset-exploration scripts (not part
                          of any prediction pipeline).
```

Each directory has its own README with full detail; this file is the map.

## Which runs were submitted

- **Subtask 1 (test split)**: `pipeline_baseline/predictions_task1_test.json`
  — Qwen3-VL-32B-Instruct, 4-bit, zero-shot, guardrailed prompt.
- **Subtask 2 (test split)**: `pipeline_advisors/predictions_task2_test.json`
  — the same decider, routed per-sample to a specialized advisor
  (ChartGemma for figures, PaddleOCR for tables), falling back to
  `pipeline_baseline`'s dual-prompt result for the samples not reached
  before the host's GPUs were reallocated (see `pipeline_advisors/README.md`
  "History" for the full account).

Dev-split scores (labelled, for local validation) are in
`pipeline_baseline/eval_results_dev.txt`: Subtask 1 accuracy 80.46 / pair
accuracy 65.34; Subtask 2 accuracy 67.61.

## Setup

```bash
uv sync
```

This creates the shared `.venv` used by `pipeline_baseline`,
`pipeline_advisors` (except its PaddleOCR advisor, which needs an isolated
venv — see `pipeline_advisors/README.md`), `evaluation_pipeline`, and
`pipeline_sibils`.

The dataset itself isn't part of this repo. `evaluation_pipeline/data/{dev,test}`
are expected to be symlinks to wherever you've downloaded the official
SciClaimEval dev/test releases (see `evaluation_pipeline/README.md` for the
expected directory shape).

## Reproducing a run

```bash
cd pipeline_baseline
./run_task1_test.sh   # -> predictions_task1_test.json
./run_task2_test.sh   # -> predictions_task2_test.json (no advisor)
./run_dev_eval.sh     # dev predictions + scoring, for validation
```

See `pipeline_advisors/README.md` for the advisor-augmented Subtask 2 run
(extraction pass, then per-advisor decision pass, then the final merge by
evidence type).

## Environment

All GPU work here targets Qwen3-VL-32B-Instruct loaded in 4-bit via
bitsandbytes, split across 2 GPUs (transformers `device_map`, not vLLM
tensor-parallelism — this host's GPUs are virtualized (GRID vGPU) slices
that reject vLLM's NCCL-based multi-GPU init). See `pipeline_baseline/README.md`
"Model & backends" for the full rationale and the alternate backends
(plain transformers, vLLM+AWQ) also supported by `qwen_agent.py`.
