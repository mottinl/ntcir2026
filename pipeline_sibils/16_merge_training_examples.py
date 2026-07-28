#!/usr/bin/env python3
"""Step 2: merges the table training examples
(`07_build_training_examples.py` -> `training_task{1,2}_tables.jsonl`) and
figure training examples (`15_build_figure_training_examples.py` ->
`training_task{1,2}_figures.jsonl`) into a single combined set, consumed by
`08_finetune_qwen3vl8b.py`/`09_eval_finetuned_peerj.py` (both are already
agnostic to the `evi_type` field -- no adaptation needed on that side).

Simple concatenation (same schema on both sides, `claim_id_pair` already
guaranteed collision-free -- `fig_` prefix on the figures side, see
`15_build_figure_training_examples.py`). Always rebuilt from scratch ("w",
not append) like the rest of the pipeline -- rerun 07 then 15 before this
one if either source changed.

Usage:
    python 16_merge_training_examples.py
"""

import argparse
from pathlib import Path


def _concat(paths: list[Path], out_path: Path) -> int:
    n = 0
    with out_path.open("w") as out:
        for p in paths:
            if not p.exists():
                print(f"  (missing, skipped: {p})")
                continue
            for line in p.read_text().split("\n"):
                if not line.strip():
                    continue
                out.write(line + "\n")
                n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-task1", default="training_task1_tables.jsonl")
    parser.add_argument("--tables-task2", default="training_task2_tables.jsonl")
    parser.add_argument("--figures-task1", default="training_task1_figures.jsonl")
    parser.add_argument("--figures-task2", default="training_task2_figures.jsonl")
    parser.add_argument("--out-task1", default="training_task1.jsonl")
    parser.add_argument("--out-task2", default="training_task2.jsonl")
    args = parser.parse_args()

    n1 = _concat([Path(args.tables_task1), Path(args.figures_task1)], Path(args.out_task1))
    n2 = _concat([Path(args.tables_task2), Path(args.figures_task2)], Path(args.out_task2))

    print(f"task1: {n1} examples -> {args.out_task1}")
    print(f"task2: {n2} examples -> {args.out_task2}")


if __name__ == "__main__":
    main()
