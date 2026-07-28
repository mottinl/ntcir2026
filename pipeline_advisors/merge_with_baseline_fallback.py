#!/usr/bin/env python3
"""Fills gaps in a (possibly interrupted, partial) advisor prediction file
with pipeline_baseline's no-advisor predictions, producing a complete
submission-ready file even if the advisor run was stopped early.

For each id in the baseline file: use the advisor prediction if present,
otherwise fall back to the baseline prediction for that same id. Never
drops an id, never fabricates one not in the baseline.

Usage:
    python merge_with_baseline_fallback.py --task task1 \
        --advisor predictions_task1_test_chartgemma.json \
        --baseline predictions_task1_test_no_advisor.json \
        --output predictions_task1_test_chartgemma_merged.json

    python merge_with_baseline_fallback.py --task task2 \
        --advisor predictions_task2_test_paddleocr.json \
        --baseline predictions_task2_test_no_advisor.json \
        --output predictions_task2_test_paddleocr_merged.json
"""

import argparse
import json
from pathlib import Path

ID_FIELD = {"task1": "claim_id", "task2": "sample_id"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["task1", "task2"])
    parser.add_argument("--advisor", required=True, help="Partial (or complete) advisor prediction file")
    parser.add_argument("--baseline", required=True, help="Complete no-advisor baseline prediction file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    id_field = ID_FIELD[args.task]

    baseline_rows = json.loads(Path(args.baseline).read_text())
    advisor_rows = json.loads(Path(args.advisor).read_text()) if Path(args.advisor).exists() else []
    advisor_by_id = {r[id_field]: r for r in advisor_rows}

    merged = []
    n_from_advisor, n_from_baseline = 0, 0
    for row in baseline_rows:
        rid = row[id_field]
        if rid in advisor_by_id:
            merged.append(advisor_by_id[rid])
            n_from_advisor += 1
        else:
            merged.append(row)
            n_from_baseline += 1

    Path(args.output).write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"Merged {len(merged)} predictions -> {args.output} "
          f"({n_from_advisor} from {args.advisor}, {n_from_baseline} from baseline fallback)")


if __name__ == "__main__":
    main()
