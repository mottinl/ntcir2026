#!/usr/bin/env python3
"""Builds ONE final task2 prediction file that routes each sample to the
advisor specialized for its evi_type: PaddleOCR for tables, ChartGemma for
figures. Falls back to pipeline_baseline's dual-prompt result for any sample not
yet covered by the relevant advisor run (partial/interrupted run).

Usage:
    python merge_specialized_by_evi_type.py --output predictions_task2_test_specialized.json
"""

import argparse
import json
from pathlib import Path

RELEASE_FILE = Path("/data/ntcir_data_test/data/test_task2_release.json")
FALLBACK_FILE = Path(__file__).resolve().parent.parent / "pipeline_baseline" / "predictions_task2_test.json"
CHARTGEMMA_FILE = Path("predictions_task2_test_chartgemma.json")
PADDLEOCR_FILE = Path("predictions_task2_test_paddleocr.json")


def load_by_id(path: Path) -> dict:
    return {r["sample_id"]: r for r in json.loads(path.read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    evi_type_by_id = {r["sample_id"]: r["evi_type"] for r in json.loads(RELEASE_FILE.read_text())}
    fallback = load_by_id(FALLBACK_FILE)
    chartgemma = load_by_id(CHARTGEMMA_FILE)
    paddleocr = load_by_id(PADDLEOCR_FILE)

    merged = []
    n_chartgemma, n_paddleocr, n_fallback = 0, 0, 0
    for sample_id, evi_type in evi_type_by_id.items():
        if evi_type == "figure" and sample_id in chartgemma:
            merged.append(chartgemma[sample_id])
            n_chartgemma += 1
        elif evi_type == "table" and sample_id in paddleocr:
            merged.append(paddleocr[sample_id])
            n_paddleocr += 1
        else:
            merged.append(fallback[sample_id])
            n_fallback += 1

    Path(args.output).write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"Merged {len(merged)} predictions -> {args.output} "
          f"({n_chartgemma} from chartgemma/figure, {n_paddleocr} from paddleocr/table, "
          f"{n_fallback} from dualprompt fallback)")


if __name__ == "__main__":
    main()
