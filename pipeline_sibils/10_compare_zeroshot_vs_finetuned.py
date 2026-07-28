#!/usr/bin/env python3
"""Compares, answer by answer, a fine-tuned prediction set against zero-shot
Qwen3-VL-8B (same base model) on the peerj dev subset, and breaks the result
down by `evi_type` (table vs figure). Automates the manual analysis done for
the first fine-tuning run (v1): overall accuracy alone masked the fact that
fine-tuning had degraded the model on its own target domain (tables) while
developing an output bias that artificially inflated the score on figures
(never seen during training).

Usage:
    python 10_compare_zeroshot_vs_finetuned.py --finetuned-preds predictions_task1_peerj_finetuned_v2.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_ZEROSHOT_PREDS = "predictions_task1_dev_8b_zeroshot_peerj_subset.json"
DEFAULT_GT = "dev_task1_release_peerj.json"


def load_preds(path: str) -> dict[str, str]:
    return {r["claim_id"]: r["pred_label"] for r in json.loads(Path(path).read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned-preds", required=True)
    parser.add_argument("--zeroshot-preds", default=DEFAULT_ZEROSHOT_PREDS)
    parser.add_argument("--gt", default=DEFAULT_GT)
    args = parser.parse_args()

    gt_records = json.loads(Path(args.gt).read_text())
    gt = {r["claim_id"]: r["label"] for r in gt_records}
    evi_type = {r["claim_id"]: r.get("evi_type") for r in gt_records}

    zs = load_preds(args.zeroshot_preds)
    ft = load_preds(args.finetuned_preds)

    ids = sorted(gt)
    missing_zs = [c for c in ids if c not in zs]
    missing_ft = [c for c in ids if c not in ft]
    if missing_zs or missing_ft:
        print(f"WARNING: missing predictions -- zero-shot: {len(missing_zs)}, fine-tuned: {len(missing_ft)}")
    ids = [c for c in ids if c in zs and c in ft]

    both_correct = both_wrong = improved = regressed = 0
    reg_by_type, imp_by_type = Counter(), Counter()
    for cid in ids:
        truth = gt[cid]
        zs_ok, ft_ok = zs[cid] == truth, ft[cid] == truth
        if zs_ok and ft_ok:
            both_correct += 1
        elif not zs_ok and not ft_ok:
            both_wrong += 1
        elif zs_ok and not ft_ok:
            regressed += 1
            reg_by_type[evi_type[cid]] += 1
        else:
            improved += 1
            imp_by_type[evi_type[cid]] += 1

    zs_acc = sum(1 for c in ids if zs[c] == gt[c]) / len(ids)
    ft_acc = sum(1 for c in ids if ft[c] == gt[c]) / len(ids)

    print("=" * 60)
    print(f"Zero-shot vs fine-tuned -- {len(ids)} examples ({args.gt})")
    print("=" * 60)
    print(f"zero-shot accuracy:  {zs_acc:.4f}")
    print(f"fine-tuned accuracy: {ft_acc:.4f}")
    print()
    print(f"both correct:                                {both_correct}")
    print(f"both wrong:                                   {both_wrong}")
    print(f"correct -> wrong after fine-tuning (regressed): {regressed}")
    print(f"wrong -> correct after fine-tuning (improved):  {improved}")
    print(f"net change in # correct: {improved - regressed:+d}")

    for et in sorted(set(evi_type.values())):
        et_ids = [c for c in ids if evi_type[c] == et]
        if not et_ids:
            continue
        zs_et_acc = sum(1 for c in et_ids if zs[c] == gt[c]) / len(et_ids)
        ft_et_acc = sum(1 for c in et_ids if ft[c] == gt[c]) / len(et_ids)
        label_dist = Counter(gt[c] for c in et_ids)
        always_majority = max(label_dist.values()) / len(et_ids)
        print()
        print(f"--- evi_type={et} ({len(et_ids)} examples, ground truth {dict(label_dist)}) ---")
        print(f"  zero-shot:  {zs_et_acc:.4f}")
        print(f"  fine-tuned: {ft_et_acc:.4f}  (delta {ft_et_acc - zs_et_acc:+.4f})")
        print(f"  regressed: {reg_by_type[et]}, improved: {imp_by_type[et]}")
        print(f"  ['always predict majority class' baseline: {always_majority:.4f}] -- "
              f"if fine-tuned accuracy is close to this, the model likely learned an output-label "
              f"bias rather than the task (this is exactly what happened with the v1 run's figures score).")

    print()
    print(f"zero-shot pred label distribution:  {Counter(zs.values())}")
    print(f"fine-tuned pred label distribution: {Counter(ft.values())}")
    print(f"ground truth label distribution:    {Counter(gt.values())}")


if __name__ == "__main__":
    main()
