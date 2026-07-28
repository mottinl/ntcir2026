#!/usr/bin/env python3
"""Step 2: lightly normalizes each sentence kept from figure_claims.jsonl
into a self-contained claim -- same logic as 04_normalize_claims.py for
tables, adapted to figure vocabulary.

`--shard i/n`: lets n instances run in parallel (one per GPU, via
CUDA_VISIBLE_DEVICES) on disjoint subsets of the remaining work, each
writing to its own file via `--output` (avoids any risk of concurrent
writes to the same file -- merge the output files afterward). The
remaining work is split by interleaving (`todo[i::n]`), not by contiguous
range, to balance the load even if one batch of figures is structurally
slower than another.

Usage:
    python 14_normalize_figure_claims.py [--limit N] [--shard i/n] [--output PATH]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline_baseline"))
from qwen_agent import QwenVLAgent, strip_thinking  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

INPUT_PATH = Path(__file__).parent / "figure_claims.jsonl"
OUTPUT_PATH = Path(__file__).parent / "figure_claims_normalized.jsonl"

PROMPT_TEMPLATE = """You are cleaning up a sentence extracted from a scientific paper so it can \
stand alone as a claim, for a dataset that pairs claims with the figure they describe.

Paragraph (context):
{context}

Extracted sentence (from the paragraph above, cites {figure_label}):
{sentence}

Task: rewrite the extracted sentence as a single self-contained claim.
- Keep it as close as possible to the original wording -- this is light \
cleanup, NOT a rewrite. Do not paraphrase, do not add information, do not \
change the meaning.
- If the sentence ends with a bare citation like "(Figure 2)" or "[Fig. 2]" \
that adds no content, you may drop it, but keep phrasing like "Figure 2 \
shows that ..." if the figure reference is part of the sentence's own claim.
- If the sentence uses a pronoun or vague reference ("it", "this", "these \
results") whose antecedent is only in the paragraph, resolve it using the \
paragraph -- but only if you can do so with high confidence.
- If the extracted sentence is truncated, garbled, or does not make sense as \
a standalone claim even with the paragraph for context, output exactly: \
SKIP

Output ONLY the final claim sentence (or SKIP), nothing else."""


def build_prompt(row: dict) -> str:
    return PROMPT_TEMPLATE.format(
        context=row["context"],
        sentence=row["sentence"],
        figure_label=row["figure_label"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N rows (for a quick QC pass)")
    parser.add_argument("--shard", type=str, default=None,
                         help="i/n -- process only every n-th remaining item, starting at "
                              "index i (0-based). For running several instances in parallel "
                              "on different GPUs without overlapping work.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                         help="Output path -- use a distinct one per shard to avoid "
                              "concurrent writes, then merge afterward.")
    args = parser.parse_args()

    # str.splitlines() wrongly breaks on real U+2028/U+2029 characters
    # present in some text fields (same bug found in 04_normalize_claims.py
    # on table_columns) -- strict split here as a precaution.
    rows = [json.loads(l) for l in INPUT_PATH.read_text().split("\n") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} candidate claims to normalize")

    # "done" is read from the *main* OUTPUT_PATH (not args.output) so every
    # shard sees the same already-normalized set, even though each shard
    # writes its own new work to a separate file.
    done: set[tuple[str, str]] = set()
    if OUTPUT_PATH.exists():
        for line in OUTPUT_PATH.read_text().split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["pmcid"], r["sentence"]))
        print(f"Resuming: {len(done)} already normalized in {OUTPUT_PATH}")

    todo = [r for r in rows if (r["pmcid"], r["sentence"]) not in done]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        todo = todo[i::n]
        print(f"Shard {i}/{n}: {len(todo)} items in this batch")
    print(f"{len(todo)} remaining to process")

    if todo:
        agent = QwenVLAgent(model_name=MODEL_NAME, max_new_tokens=256)

    n_skipped = 0
    with args.output.open("a") as f:
        for i, row in enumerate(todo):
            prompt = build_prompt(row)
            raw = agent.generate([], prompt, greedy=True)
            answer = strip_thinking(raw).strip()
            answer = re.sub(r'^["\']|["\']$', "", answer).strip()

            if answer.upper() == "SKIP" or not answer:
                n_skipped += 1
                print(f"[{i+1}/{len(todo)}] {row['pmcid']} -> SKIP")
                continue

            print(f"[{i+1}/{len(todo)}] {row['pmcid']} -> {answer[:100]}")
            out_row = dict(row)
            out_row["claim"] = answer
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()

    n_kept = len(todo) - n_skipped
    print(f"\nThis run: {n_kept}/{len(todo)} normalized, {n_skipped} SKIP")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
