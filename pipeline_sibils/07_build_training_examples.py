#!/usr/bin/env python3
"""Step 1: assembles the final training files from `perturbed_claims.jsonl`
+ the PNGs already rendered under `/data/pipeline_sibils_cache/tables_png/`
(`06_render_table_png.py`).

Two outputs, in a format close to the official schema (same field names as
`dev_task{1,2}_release.json`, so it's directly reusable by a future
fine-tuning/eval script with no further transformation):

- `training_task1_tables.jsonl`: one Supported/Refuted pair per source row
  (same claim, original vs. perturbed `evi_path`), linked by
  `claim_id_pair`.
- `training_task2_tables.jsonl`: one evidence pair (original + perturbed)
  per source row, evidence_id_1/evidence_id_2 assignment randomized (a
  deterministic seed per `(pmcid, idx)`) so a future model isn't biased
  toward "the right answer is always evidence_id_1".

`_tables` suffix: see `15_build_figure_training_examples.py` (same schema,
figures) and `16_merge_training_examples.py`, which combines the two into
`training_task{1,2}.jsonl` -- the final file actually consumed by
`08_finetune_qwen3vl8b.py`/`09_eval_finetuned_peerj.py`.

v3: the "Caption adjustments" operation (05_perturb_table.py) keeps the
table unchanged and perturbs the caption instead -- so the Refuted entry
uses `row["perturbed_caption"]` instead of `row["table_caption"]` for task1
(same image pixels as Supported, different caption). These rows are
excluded from task2: both evidence_id options would point to strictly
identical images, a choice this format can't represent (no per-option
caption field).

Usage:
    python 07_build_training_examples.py
"""

import argparse
import hashlib
import json
from pathlib import Path

INPUT_PATH = Path(__file__).parent / "perturbed_claims.jsonl"
PNG_DIR = Path("/data/pipeline_sibils_cache/tables_png")


def _seeded_bool(pmcid: str, idx: int) -> bool:
    """Deterministic pseudo-random bit from (pmcid, idx), so evidence_id_1
    isn't always the supported one across the whole file."""
    h = hashlib.sha256(f"{pmcid}_{idx}".encode()).digest()
    return h[0] % 2 == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-task1", default="training_task1_tables.jsonl")
    parser.add_argument("--out-task2", default="training_task2_tables.jsonl")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--png-dir", type=Path, default=PNG_DIR)
    args = parser.parse_args()

    # str.splitlines() wrongly breaks on real U+2028/U+2029 characters
    # found inside a table cell (e.g. PMC2874376: "MLST<U+2028>ST") -- strict
    # split here (same fix as in 04_normalize_claims.py).
    rows = [json.loads(l) for l in args.input.read_text().split("\n") if l.strip()]

    task1_out, task2_out = [], []
    missing_png = 0

    for idx, row in enumerate(rows):
        pmcid = row["pmcid"]
        sup_png = args.png_dir / f"{pmcid}_{idx}_supported.png"
        ref_png = args.png_dir / f"{pmcid}_{idx}_refuted.png"
        if not (sup_png.exists() and ref_png.exists()):
            missing_png += 1
            continue

        claim_id_pair = f"{pmcid}_{idx}"
        sup_caption = row.get("table_caption") or ""
        ref_caption = row.get("perturbed_caption") or sup_caption
        is_caption_op = row.get("operation") == "Caption adjustments"

        common = dict(
            claim=row["claim"],
            evi_type="table",
            context=row.get("context") or "",
            domain="peerj_sibils",
            claim_id_pair=claim_id_pair,
        )

        task1_out.append({
            "claim_id": f"{claim_id_pair}_s",
            **common,
            "caption": sup_caption,
            "label": "Supported",
            "operation": "Supported_claim_only",
            "evi_path": str(sup_png),
        })
        task1_out.append({
            "claim_id": f"{claim_id_pair}_r",
            **common,
            "caption": ref_caption,
            "label": "Refuted",
            "operation": row.get("operation") or "Change the cell values",
            "evi_path": str(ref_png),
        })

        if is_caption_op:
            # evidence_id_1/2 would point to byte-identical images -- task2
            # has no per-option caption field, so this operation can't be
            # represented here (see module docstring).
            continue

        id1_is_supported = _seeded_bool(pmcid, idx)
        evidence_id_1 = str(sup_png if id1_is_supported else ref_png)
        evidence_id_2 = str(ref_png if id1_is_supported else sup_png)
        task2_out.append({
            "sample_id": claim_id_pair,
            "claim": row["claim"],
            "caption": sup_caption,
            "context": row.get("context") or "",
            "evi_type": "table",
            "domain": "peerj_sibils",
            "evidence_id_1": evidence_id_1,
            "evidence_id_2": evidence_id_2,
            "label": "evidence_id_1" if id1_is_supported else "evidence_id_2",
        })

    Path(args.out_task1).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in task1_out) + "\n"
    )
    Path(args.out_task2).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in task2_out) + "\n"
    )

    print(f"task1: {len(task1_out)} examples ({len(task1_out)//2} pairs) -> {args.out_task1}")
    print(f"task2: {len(task2_out)} examples -> {args.out_task2}")
    if missing_png:
        print(f"skipped {missing_png} source rows with missing PNG(s)")

    label1_dist = sum(1 for r in task2_out if r["label"] == "evidence_id_1")
    print(f"task2 label balance: evidence_id_1={label1_dist}, evidence_id_2={len(task2_out)-label1_dist}")


if __name__ == "__main__":
    main()
