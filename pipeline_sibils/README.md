# pipeline_sibils — synthetic training data from SIBiLS + QLoRA fine-tuning

A separate experiment from `pipeline_baseline`/`pipeline_advisors`: instead
of relying purely on zero-shot prompting, this pipeline builds a synthetic
SciClaimEval-style training set from biomedical literature indexed by
[SIBiLS](https://www.sibils.org/), reproducing the task organizers' own
claim/evidence-pair construction recipe for the dataset's `peerj`
sub-domain, and fine-tunes Qwen3-VL-8B-Instruct on it with QLoRA.

**Status**: exploratory. This pipeline did not produce the runs submitted
to the shared task -- `pipeline_baseline` and `pipeline_advisors` did. It's
kept here as a documented, self-contained experiment: the data-construction
half (steps 0-2) works end-to-end and produces a usable synthetic dataset;
the fine-tuning half (steps 3-4) ran to completion but did not surpass the
zero-shot baseline on held-out `peerj` data. The code is left as-is for
reference or as a starting point for a different fine-tuning recipe.

## Why this exists

The official dataset has no `train` split. `dev` (747 labelled examples,
domains `nlp`/`ml`/`peerj`) and the unlabelled `test` split are all that's
released. `peerj` (the biomedical domain, ~26% of dev) traces back to PMC,
which SIBiLS indexes with structured tables already parsed out of the
article body -- making it feasible to construct additional
claim/evidence pairs for that one domain without needing a table/figure
parser from scratch. `nlp`/`ml` (arXiv/ACL papers) are out of reach of
this approach and out of scope here.

## Pipeline stages

The scripts are numbered in the order they run. Each is resumable (keyed
JSONL append) and safe to re-run.

**Step 0 — corpus discovery** (`00_exclusion_list.py`, `00_test_fetch.py`,
`01_query_sibils.py`): builds the anti-leakage exclusion list (every
`peerj` paper already in dev/test, resolved from PeerJ's own article number
to a real PMCID via NCBI's DOI converter -- the two are unrelated numbers),
explores the SIBiLS `fetch`/`search` API, and queries ~1000 candidate
biomedical PMCIDs across three topical buckets (general biomed, biomed+ML,
biomed+NLP), filtered by license and by the presence of real tables
(`tables_in_body > 0`).

**Step 1 — table claims** (`02_fetch_candidates.py` through
`07_build_training_examples.py`): fetches each candidate's full parsed
document from SIBiLS; extracts sentences that cite a table by regex/
heuristic (no LLM at this stage); lightly normalizes each sentence into a
self-contained claim with Qwen3-VL-8B in text-only mode
(`04_normalize_claims.py`, with a CPU-only and a vLLM-batched variant for
different resource situations); perturbs the table's cell values (and,
later, row/column swaps, whole-column scaling, and caption-only edits, to
better match the official operation taxonomy) to generate the Refuted
counterpart; renders both the original and perturbed table to a PNG
matching the official dataset's visual style; and assembles the final
`training_task{1,2}_tables.jsonl` files.

**Step 2 — figure claims** (`11_resolve_pmc_oa.py` through
`16_merge_training_examples.py`): SIBiLS itself never exposes binary
images, so figures are downloaded separately from the public PMC Open
Access S3 bucket; sentences citing a figure are extracted the same way as
for tables; figures are perturbed via **Graph Swap** (reordering panels of
a multi-panel figure, detected by whitespace-gutter analysis) or
**Category Swap** (swapping two existing OCR-detected text labels), the two
operations that cover ~76% of the official dataset's real figure edits and
that can be done as pure pixel manipulation, without needing the
underlying chart data; and the table and figure training sets are merged
into `training_task{1,2}.jsonl`.

**Step 3 — fine-tuning** (`08_finetune_qwen3vl8b.py`): QLoRA (4-bit NF4
base, LoRA on the language model's attention/MLP projections, vision tower
untouched) on `training_task1.jsonl`, using a deliberately simple prompt
(not `pipeline_baseline`'s guardrailed Rec-3 prompt, which exists to
compensate zero-shot for a non-fine-tuned model's blind spots).
`17_check_training_image_footprint.py` is a pre-flight check on visual-token
load, added after diagnosing a persistent CUDA OOM traced back to the
heaviest perturbation operation on this project's virtualized GPU.

**Step 4 — evaluation** (`09_eval_finetuned_peerj.py`,
`10_compare_zeroshot_vs_finetuned.py`, `18_predict_test.py`): scores the
fine-tuned adapter against `pipeline_baseline`'s zero-shot predictions,
both restricted to the dev set's `peerj` subset for an apples-to-apples
comparison (pipeline_baseline's overall scores are diluted by `nlp`/`ml`,
which this pipeline never trains on); breaks the comparison down by
`evi_type` to check for a labelling bias rather than a genuine task
improvement; and generates (unlabelled) predictions on the official
`peerj` test subset with and without the adapter, for reference.

## Result

Across several perturbation-taxonomy iterations, fine-tuning on this
synthetic corpus did not improve on the zero-shot baseline for the trained
domain (`peerj`, tables) on held-out dev data -- the recurring failure
pattern was a model that fits the synthetic training distribution well
(near-zero train loss) without transferring that to genuinely verifying a
table against a claim. Whether this reflects a ceiling of the perturbation-based
label-construction approach itself, insufficient data/diversity, or a
fixable training/prompt issue is left open; the code and this account are
kept as a documented, neutral record of the attempt rather than a
recommendation for or against the approach.

## SIBiLS API notes

- REST endpoints: `https://biodiversitypmc.sibils.org/api/{search,fetch}`.
  `fetch?ids=<PMCID,...>&col=pmc` (up to 1000 ids/call) returns
  `{sibils_article_set: [{document: {...}}]}` per PMCID; `document` holds
  metadata (`pmcid`, `doi`, `title`, `licence`, `tables_in_body`,
  `figures_in_body`) and `body_sections` (a list of sections, each with
  `contents[i].tag` in `{"p", "list-item", "table", "fig"}`).
- Tables (`tag=="table"`) arrive already parsed: `table_columns`,
  `table_values` (a 2D array), `caption`, `label` (e.g. "Table 1"),
  `xref_id` -- no HTML to parse.
- Figures (`tag=="fig"`) only carry `caption`, `label`, and `graphics` (a
  bare filename, no binary and no URL) -- the actual image bytes have to
  come from elsewhere (this pipeline uses the PMC Open Access S3 bucket,
  matching on that same filename).
- `search?jq=<Elasticsearch JSON>&col=pmc&n=<count>` gives raw ES hits,
  filterable server-side (e.g. a `terms` filter on `licence`).
- A direct Elasticsearch index also exists for bulk export beyond the REST
  API's 1000-id cap, not used here (this corpus is a few thousand papers at
  most).

## Requirements

Uses the shared repo-root `.venv` for everything except figure perturbation
(`14_perturb_figures.py` needs `easyocr`, not in the default dependency
set). No isolated venv is required otherwise.
