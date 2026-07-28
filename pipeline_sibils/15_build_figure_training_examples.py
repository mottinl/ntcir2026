#!/usr/bin/env python3
"""Step 2: assembles the figure training files from
`figure_claims_normalized.jsonl` (normalized claim, 04_normalize_claims.py
adapted for figures) + `perturbed_figure_claims.jsonl` (perturbed image,
14_perturb_figures.py -- Graph Swap / Category Swap). The two are produced
independently (image perturbation doesn't depend on the LLM), joined here by
the `(pmcid, sentence)` key.

Same output schema as `07_build_training_examples.py` (same field names as
the official format), `evi_type="figure"`:
- `training_task1_figures.jsonl`: one Supported/Refuted pair per perturbed
  row, PLUS a `Supported_claim_only` entry (no Refuted) for every
  normalized claim whose image couldn't be perturbed (~19% of the figure
  corpus -- not enough swappable panels/labels). Unlike tables (where the
  fallback almost always guarantees a perturbation), the absence of a
  perturbation is frequent and expected here -- `Supported_claim_only`
  already exists in the official dataset (33/97 of the dev set's peerj
  figures), so keeping these claims as pure positives instead of dropping
  them is faithful to the real distribution, not wasted data.
- `training_task2_figures.jsonl`: one evidence pair per perturbed row only
  (`Supported_claim_only` entries have no second evidence to pair with,
  like the "Caption adjustments" rows on the table side).

`claim_id_pair` is prefixed `fig_` (unlike tables' raw `pmcid_idx`) to
guarantee no collision once the two files are merged (the same pmcid can
contribute both a table claim and a figure claim, potentially with the same
numeric idx in both source files).

Usage:
    python 15_build_figure_training_examples.py
"""

import argparse
import hashlib
import json
from pathlib import Path

NORMALIZED_PATH = Path(__file__).parent / "figure_claims_normalized.jsonl"
PERTURBED_PATH = Path(__file__).parent / "perturbed_figure_claims.jsonl"


def _seeded_bool(pmcid: str, idx: int) -> bool:
    """Deterministic pseudo-random bit from (pmcid, idx), so evidence_id_1
    isn't always the supported one across the whole file."""
    h = hashlib.sha256(f"fig_{pmcid}_{idx}".encode()).digest()
    return h[0] % 2 == 0


def _load_jsonl(path: Path) -> list[dict]:
    # str.splitlines() wrongly breaks on real U+2028/U+2029 characters
    # (same fix as in 04_normalize_claims.py) -- strict split here.
    return [json.loads(l) for l in path.read_text().split("\n") if l.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-task1", default="training_task1_figures.jsonl")
    parser.add_argument("--out-task2", default="training_task2_figures.jsonl")
    parser.add_argument("--normalized", type=Path, default=NORMALIZED_PATH)
    parser.add_argument("--perturbed", type=Path, default=PERTURBED_PATH)
    args = parser.parse_args()

    normalized_rows = _load_jsonl(args.normalized)
    perturbed_by_key = {(r["pmcid"], r["sentence"]): r for r in _load_jsonl(args.perturbed)}
    print(f"{len(normalized_rows)} normalized claims, "
          f"{len(perturbed_by_key)} with a perturbation available")

    task1_out, task2_out = [], []
    n_paired = 0
    n_supported_only = 0
    n_missing_image = 0

    for idx, row in enumerate(normalized_rows):
        key = (row["pmcid"], row["sentence"])
        pmcid = row["pmcid"]
        claim_id_pair = f"fig_{pmcid}_{idx}"
        caption = row.get("figure_caption") or ""
        common = dict(
            claim=row["claim"],
            evi_type="figure",
            context=row.get("context") or "",
            domain="peerj_sibils",
            claim_id_pair=claim_id_pair,
        )

        perturbed = perturbed_by_key.get(key)
        orig_path = Path(row["local_path"])
        if not orig_path.exists():
            n_missing_image += 1
            continue

        task1_out.append({
            "claim_id": f"{claim_id_pair}_s",
            **common,
            "caption": caption,
            "label": "Supported",
            "operation": "Supported_claim_only",
            "evi_path": str(orig_path),
        })

        if perturbed is None:
            # No swappable panel/label found for this image (Graph Swap /
            # Category Swap both inapplicable) -- keep the Supported-only
            # entry, no Refuted counterpart (see module docstring).
            n_supported_only += 1
            continue

        ref_path = Path(perturbed["perturbed_local_path"])
        if not ref_path.exists():
            n_missing_image += 1
            continue

        n_paired += 1
        task1_out.append({
            "claim_id": f"{claim_id_pair}_r",
            **common,
            "caption": caption,
            "label": "Refuted",
            "operation": perturbed["operation"],
            "evi_path": str(ref_path),
        })

        id1_is_supported = _seeded_bool(pmcid, idx)
        evidence_id_1 = str(orig_path if id1_is_supported else ref_path)
        evidence_id_2 = str(ref_path if id1_is_supported else orig_path)
        task2_out.append({
            "sample_id": claim_id_pair,
            "claim": row["claim"],
            "caption": caption,
            "context": row.get("context") or "",
            "evi_type": "figure",
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

    print(f"task1: {len(task1_out)} examples ({n_paired} paires Supported/Refuted + "
          f"{n_supported_only} Supported_claim_only) -> {args.out_task1}")
    print(f"task2: {len(task2_out)} examples -> {args.out_task2}")
    if n_missing_image:
        print(f"{n_missing_image} rows skipped (image missing on disk)")

    label1_dist = sum(1 for r in task2_out if r["label"] == "evidence_id_1")
    print(f"task2 label balance: evidence_id_1={label1_dist}, evidence_id_2={len(task2_out)-label1_dist}")


if __name__ == "__main__":
    main()
