# analysis — dataset exploration

Two standalone scripts used early on to understand the SciClaimEval dataset
before designing `pipeline_baseline`'s prompts. Not part of the prediction
pipelines themselves.

- `data_analysis.py` — descriptive stats on the test split (task1 + task2):
  field inventory, claims-per-paper, domain/evidence-type/license
  breakdowns, claim/caption/context length distributions, and a task1-vs-
  task2 paper overlap check. Pure stdlib, no GPU. Output: `data_analysis_output.txt`.
- `figures_analysis.py` — classifies every figure-type evidence image with
  Qwen3-VL-32B (dominant chart type + panel count), to estimate how many
  figures are actually composite multi-panel images. This matters for
  evaluation since a claim may refer to only one panel out of several.
  Its 2-GPU bitsandbytes 4-bit loading approach (`build_device_map`) is
  where `pipeline_baseline/qwen_agent.py`'s equivalent was first validated.
  Output: `figures_classification_<split>_qwen3vl32b.json`.

## Running

```bash
cd analysis
source ../.venv/bin/activate
python data_analysis.py
python figures_analysis.py --split dev --limit 10   # smoke test
python figures_analysis.py --split dev
```

`figures_classification_qwen3vl.json` is an earlier classification run
(different model/config) kept for reference alongside the current
`_dev_qwen3vl32b` one.
