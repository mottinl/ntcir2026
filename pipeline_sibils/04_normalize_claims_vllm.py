#!/usr/bin/env python3
"""Step 1: vLLM (batched) variant of 04_normalize_claims.py, for a future
larger-scale run if the candidate corpus is expanded.

04_normalize_claims.py remains the reference record -- deliberately left
unmodified -- of what actually ran and produced claims_normalized.jsonl
(1069/1080). This script has NOT yet been run under real conditions; it's
ready for the next time a larger volume of sentences needs normalizing.

Motivation: the reference run called agent.generate() one sentence at a
time (~2.6s/item observed), plain transformers backend, no batching at all
-- the GPU stayed under-utilized at batch=1 for completions this short (max
256 tokens). Here Qwen/Qwen3-VL-8B-Instruct (bf16 checkpoint, not AWQ) is
routed through vLLM via qwen_agent.QwenVLAgent(backend="vllm"), and prompts
are submitted in batches to agent.generate_batch(), which benefits from
vLLM's continuous batching instead of a sequential loop -- see qwen_agent.py
for the batched implementation's details.

The prompt and the resume logic (the (pmcid, sentence) key already written
to claims_normalized.jsonl) are duplicated from 04_normalize_claims.py
rather than imported, so as not to touch that reference file (and because
importing a module whose name starts with a digit needs importlib anyway,
not a plain import).

Usage:
    python 04_normalize_claims_vllm.py [--limit N] [--batch-size N]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline_baseline"))
from qwen_agent import QwenVLAgent, strip_thinking  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

INPUT_PATH = Path(__file__).parent / "table_mentions.jsonl"
OUTPUT_PATH = Path(__file__).parent / "claims_normalized.jsonl"

# Identical to 04_normalize_claims.py -- see that file for the rationale
# behind each instruction (light cleanup, not an aggressive rewrite).
PROMPT_TEMPLATE = """You are cleaning up a sentence extracted from a scientific paper so it can \
stand alone as a claim, for a dataset that pairs claims with the table they describe.

Paragraph (context):
{context}

Extracted sentence (from the paragraph above, cites {table_label}):
{sentence}

Task: rewrite the extracted sentence as a single self-contained claim.
- Keep it as close as possible to the original wording -- this is light \
cleanup, NOT a rewrite. Do not paraphrase, do not add information, do not \
change the meaning.
- If the sentence ends with a bare citation like "(Table 2)" or "[Table 2]" \
that adds no content, you may drop it, but keep phrasing like "Table 2 \
shows that ..." if the table reference is part of the sentence's own claim.
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
        table_label=row["table_label"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N rows (for a quick QC pass)")
    parser.add_argument("--batch-size", type=int, default=32,
                         help="Prompts submitted together per vLLM continuous-batching call")
    args = parser.parse_args()

    rows = [json.loads(l) for l in INPUT_PATH.read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} candidate claims to normalize")

    # Same resume key as 04_normalize_claims.py -- both scripts read/write
    # the same OUTPUT_PATH, so they're interchangeable across runs (e.g.
    # resuming with vLLM a run started on the plain transformers backend,
    # or the other way around).
    done: set[tuple[str, str]] = set()
    if OUTPUT_PATH.exists():
        for line in OUTPUT_PATH.read_text().splitlines():
            r = json.loads(line)
            done.add((r["pmcid"], r["sentence"]))
        print(f"Resuming: {len(done)} already normalized in {OUTPUT_PATH}")

    todo = [r for r in rows if (r["pmcid"], r["sentence"]) not in done]
    print(f"{len(todo)} remaining to process")

    if not todo:
        return

    agent = QwenVLAgent(model_name=MODEL_NAME, backend="vllm", max_new_tokens=256)

    n_skipped = 0
    n_kept = 0
    with OUTPUT_PATH.open("a") as f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            items = [([], build_prompt(row)) for row in batch]
            outputs = agent.generate_batch(items, greedy=True)

            for row, raw in zip(batch, outputs):
                answer = strip_thinking(raw).strip()
                # Strip surrounding quotes the model sometimes adds despite the instruction.
                answer = re.sub(r'^["\']|["\']$', "", answer).strip()

                if answer.upper() == "SKIP" or not answer:
                    n_skipped += 1
                    continue

                out_row = dict(row)
                out_row["claim"] = answer
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                n_kept += 1
            f.flush()

            done_so_far = min(i + args.batch_size, len(todo))
            print(f"[{done_so_far}/{len(todo)}] batch processed "
                  f"({n_kept} kept, {n_skipped} SKIP so far)")

    print(f"\nThis run: {n_kept}/{len(todo)} normalized, {n_skipped} SKIP")
    n_total = len(done) + n_kept
    print(f"Total so far: {n_total}/{len(rows)} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
