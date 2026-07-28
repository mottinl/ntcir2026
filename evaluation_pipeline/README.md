# evaluation_pipeline — SciClaimEval (NTCIR-19)

Workspace for Subtask 1 (Claim Label Prediction) and Subtask 2 (Claim-Evidence
Prediction). See https://sciclaimeval.github.io for the task description.

## Layout

```
evaluation_pipeline/
├── data/
│   ├── dev/   -> symlink to /data/ntcir_data_train   (dev split, has gold labels)
│   └── test/  -> symlink to /data/ntcir_data_test    (formal-run split, no gold labels)
└── eval/      vendored from github.com/SciClaimEval/sciclaimeval-shared-task
    ├── run_eval.py                   entry point, see below
    ├── evaluation/eval_script.py     scoring logic
    ├── examples/                     tiny dummy pred/gold files used by run_eval.py defaults
    ├── reproduce_all_models_task1.py
    └── reproduce_all_models_task2.py
```

`data/dev` and `data/test` are symlinks, not copies — the datasets were already
downloaded to `/data/ntcir_data_train` and `/data/ntcir_data_test`, so nothing
was re-fetched.

## Data

Both `dev` and `test` follow the same directory shape:

```
data/<split>/data/
├── <split>_task1_release.json
├── <split>_task2_release.json
├── figures/<split>/*.png
├── tables/<split>/*.tex|*.html      (original table source)
├── tables_png/<split>/*.png         (rendered table image, used as evi_path)
└── papers/<split>/*.json            (full paper text)
```

Row counts: dev task1 = 747, dev task2 = 352, test task1 = 917, test task2 = 436.
`test` is missing `label` (task1) — gold is withheld for the formal run.

Key fields (task1): `claim_id`, `claim`, `label` (`Supported`/`Refuted`),
`evi_type` (`table`/`figure`), `evi_path` (PNG), `evi_path_original` (tex/html,
tables only), `context`, `domain` (`ml`/`nlp`/`peerj`), `claim_id_pair` (links
the Supported/Refuted version of the same claim — used for pair-accuracy).

Task2 adds `sample_id`, `evidence_id_1`, `evidence_id_2`, and a fixed
`question` asking which evidence supports the claim.

## Environment

Deps live in the repo-root `pyproject.toml` / `.venv`, shared with
`pipeline_baseline` and `pipeline_advisors` (one uv workspace for the whole
repo). `pandas` and `scikit-learn` were added for the eval scripts:

```bash
cd ntcir2026
uv sync
```

## Running evaluation

`run_eval.py` must be run from inside `eval/` (it imports `evaluation.eval_script`
as a local package):

```bash
cd evaluation_pipeline/eval
python run_eval.py --task task1 \
  --ground_truth_task1 ../data/dev/data/dev_task1_release.json \
  --pred_task1 <your_predictions.json>

python run_eval.py --task task2 \
  --ground_truth_task2 ../data/dev/data/dev_task2_release.json \
  --pred_task2 <your_predictions.json>
```

Prediction format:
- task1: `[{"claim_id": "val_tab_0001", "pred_label": "Refuted"}]`
- task2: `[{"sample_id": "val_0071", "pred_label": "evidence_id_1"}]`

Metrics: task1 reports precision/recall/macro-F1/accuracy plus **pair
accuracy** (primary metric — both Supported/Refuted versions of a claim must
be correct); task2 reports plain accuracy (primary metric).

## Note: fixed upstream bug

`eval/evaluation/eval_script.py::eval_task_1_individual` had a bug in the
upstream repo (it called itself instead of `eval_task_1_individual_data`,
causing an infinite-recursion crash). Fixed locally in our vendored copy —
not yet reported/patched upstream.
