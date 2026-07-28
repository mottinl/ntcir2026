#!/usr/bin/env python3
"""Merge shard prediction files (from --num-shards runs) into one file.

Usage:
    python merge_shards.py shard0.json shard1.json -o predictions_task1_dev.json
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", help="Shard prediction JSON files")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    merged = []
    for shard_path in args.shards:
        merged.extend(json.loads(open(shard_path).read()))

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Merged {len(args.shards)} shards -> {len(merged)} records -> {args.output}")


if __name__ == "__main__":
    main()
