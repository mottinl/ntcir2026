#!/usr/bin/env python3
"""Step 1: lightly normalizes each sentence kept from table_mentions.jsonl
into a self-contained claim.

The official SciClaimEval claims are "close to verbatim from the author's
sentence" (e.g. "Table 1 shows that our probabilistic inference module
consistently outperforms..."), so this is deliberately NOT an aggressive
rewrite: it just drops a purely-referential trailing citation like
"... (Table 2)." when the table reference adds no content, resolves obvious
pronoun references using the paragraph for context, and rejects sentences
visibly truncated by 03_extract_claims.py's naive sentence splitter (e.g. a
cut on "Fig." mid-sentence).

Model: reuses Qwen/Qwen3-VL-8B-Instruct (already cached locally, see
../pipeline_baseline/README.md) in text-only mode (no image passed to
generate()) rather than downloading a separate text-only Qwen3-8B -- no
text-only model is currently cached under /data/models, and the extra ~16GB
download isn't justified here since the VL checkpoint behaves identically in
text-only mode (its tokenizer/chat template doesn't depend on images being
present in the message content).

Usage:
    python 04_normalize_claims.py [--limit N]
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
    args = parser.parse_args()

    # str.splitlines() wrongly breaks on real U+2028/U+2029 ("line
    # separator"/"paragraph separator") characters found inside a table
    # cell (e.g. PMC2874376: "MLST<U+2028>ST") -- a valid JSONL can only be
    # split on literal "\n" between records, hence the strict split here.
    rows = [json.loads(l) for l in INPUT_PATH.read_text().split("\n") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} candidate claims to normalize")

    # Resumable: (pmcid, sentence) is a stable key per input row (sentence is
    # the raw extracted text, unique enough within a paper for this pilot's
    # volume). Rebuilding "done" from the output file -- not just counting
    # output lines -- is required because SKIPped rows never get written, so
    # line count alone can't tell how many *input* rows were consumed.
    done: set[tuple[str, str]] = set()
    if OUTPUT_PATH.exists():
        for line in OUTPUT_PATH.read_text().split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["pmcid"], r["sentence"]))
        print(f"Resuming: {len(done)} already normalized in {OUTPUT_PATH}")

    todo = [r for r in rows if (r["pmcid"], r["sentence"]) not in done]
    print(f"{len(todo)} remaining to process")

    if todo:
        agent = QwenVLAgent(model_name=MODEL_NAME, max_new_tokens=256)

    n_skipped = 0
    with OUTPUT_PATH.open("a") as f:
        for i, row in enumerate(todo):
            prompt = build_prompt(row)
            raw = agent.generate([], prompt, greedy=True)
            answer = strip_thinking(raw).strip()
            # Strip surrounding quotes the model sometimes adds despite the instruction.
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
    n_total = len(done) + n_kept
    print(f"Total so far: {n_total}/{len(rows)} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
