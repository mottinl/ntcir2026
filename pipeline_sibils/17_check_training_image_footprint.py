#!/usr/bin/env python3
"""Step 3: measures the visual-token load (post-resize image area) of the
combined training set BEFORE launching a real fine-tuning run -- the same
diagnostic that, after the fact, confirmed "Full table altercations" was
the cause of the v4 run's persistent CUDA OOM, but done HERE up front
instead of after a failure.

Applies the exact same resize as `_load_image` in
`08_finetune_qwen3vl8b.py` (MAX_IMAGE_SIDE=1536, MAX_IMAGE_WIDTH=1240,
reduced from 2048/1655 after a deterministic OOM at step 210/1698 of the
tables+figures run -- see the comment in 08_finetune_qwen3vl8b.py) --
figures had never been measured from this angle before (new in the
combined set, real downloaded JPEGs, unlike tables which are rendered
in-house and whose size distribution was therefore controlled). Breaks
down by `evi_type` to see whether figures push the memory load above the
known-stable reference (v3: mean 1.145 Mpx, p90 2.382, max 3.361 -- values
measured with the OLD 2048/1655 cap, kept here as a historical reference).

Usage:
    python 17_check_training_image_footprint.py [--train-file training_task1.jsonl]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

MAX_IMAGE_SIDE = 1536
MAX_IMAGE_WIDTH = 1240


def resized_area_mpx(path: str) -> float | None:
    try:
        with Image.open(path) as img:
            w, h = img.size
    except Exception as e:
        print(f"  [!] impossible d'ouvrir {path}: {e}")
        return None
    scale = min(MAX_IMAGE_SIDE / h, MAX_IMAGE_WIDTH / w, 1.0)
    if scale < 1.0:
        w, h = w * scale, h * scale
    return (w * h) / 1_000_000


def summarize(areas: list[float]) -> dict:
    if not areas:
        return {}
    s = sorted(areas)
    n = len(s)
    return {
        "n": n,
        "mean": sum(s) / n,
        "p90": s[int(0.9 * n)] if n > 1 else s[0],
        "max": s[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="training_task1.jsonl")
    args = parser.parse_args()

    rows = [json.loads(l) for l in Path(args.train_file).read_text().split("\n") if l.strip()]
    print(f"{len(rows)} examples in {args.train_file}")

    # One image per unique evi_path (Supported/Refuted from the same table
    # pair often share a very close size, but it's not guaranteed -- this
    # measures every actually-referenced file, not just one per pair).
    areas_by_type: dict[str, list[float]] = defaultdict(list)
    seen_paths = set()
    for row in rows:
        path = row["evi_path"]
        if path in seen_paths:
            continue
        seen_paths.add(path)
        area = resized_area_mpx(path)
        if area is not None:
            areas_by_type[row["evi_type"]].append(area)

    print(f"\n{len(seen_paths)} unique image files measured\n")
    print(f"{'evi_type':<10} {'n':>6} {'mean Mpx':>10} {'p90 Mpx':>10} {'max Mpx':>10}")
    all_areas = []
    for evi_type, areas in areas_by_type.items():
        stats = summarize(areas)
        all_areas.extend(areas)
        print(f"{evi_type:<10} {stats['n']:>6} {stats['mean']:>10.3f} {stats['p90']:>10.3f} {stats['max']:>10.3f}")
    stats = summarize(all_areas)
    print(f"{'TOTAL':<10} {stats['n']:>6} {stats['mean']:>10.3f} {stats['p90']:>10.3f} {stats['max']:>10.3f}")

    print("\nKnown stable reference (v3, tables only, 2 operations):")
    print(f"{'v3 ref':<10} {'':>6} {1.145:>10.3f} {2.382:>10.3f} {3.361:>10.3f}")


if __name__ == "__main__":
    main()
