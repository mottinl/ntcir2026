#!/usr/bin/env python3
"""CPU-only variant of `04_normalize_claims.py`.

Same task, same prompt, same output file (`claims_normalized.jsonl`, resumed
by the `(pmcid, sentence)` key) -- but the model is forced onto CPU instead
of GPU. Useful for processing new rows brought in by a corpus expansion
(e.g. new documents added to `candidate_pmcids.jsonl`) without using a GPU,
in parallel with other GPU work in progress.

`QwenVLAgent` (pipeline_baseline/qwen_agent.py) is NOT reused here: its
__init__ hardcodes `device_map="auto"` (would go to GPU if available) and
`attn_implementation="flash_attention_2"` (requires a GPU). So the model is
loaded manually here, with `CUDA_VISIBLE_DEVICES=""` set before any
torch/transformers import as an extra guard (in case a device still got
resolved implicitly somewhere in the stack).

Expected: noticeably slower than the GPU version (8B params in bf16 on CPU,
no bf16 hardware acceleration on this host -- avx512_bf16 absent from the
CPU flags). Budget several seconds to ~1 minute per claim depending on
context length; run this in the background for a volume of several hundred
rows.

Important: this script runs alongside other GPU work in progress, which
itself needs some CPU (dataloader, collation, etc.). By default it
therefore deliberately takes only a fraction of the cores (not
`os.cpu_count()`) and runs at a lowered CPU priority (`nice`), so as not to
starve the other task -- adjust --threads if needed.

Usage:
    python 04_normalize_claims_bis.py [--limit N] [--threads N]
"""

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline_baseline"))
from qwen_agent import strip_thinking  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

INPUT_PATH = Path(__file__).parent / "table_mentions.jsonl"
OUTPUT_PATH = Path(__file__).parent / "claims_normalized.jsonl"

# Identical to 04_normalize_claims.py (same prompt -> same normalization
# behavior, only the compute device differs).
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


class _CpuQwenAgent:
    """Minimal loading, CPU forced, bf16 (the checkpoint's native on-disk
    dtype -- avoids a dtype conversion at load time)."""

    def __init__(self, model_name: str, max_new_tokens: int = 256):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        import torch

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == "", \
            "CUDA_VISIBLE_DEVICES must be cleared before importing torch to guarantee CPU-only"

        print(f"Loading {model_name} on CPU (bf16) ...", flush=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="cpu",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        print("Model loaded (CPU).", flush=True)

    def generate(self, text: str, greedy: bool = False) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        gen_kwargs = dict(max_new_tokens=self.max_new_tokens)
        if greedy:
            gen_kwargs.update(do_sample=False)
        else:
            gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N remaining rows (quick test)")
    default_threads = max(4, (os.cpu_count() or 4) // 4)
    parser.add_argument("--threads", type=int, default=default_threads,
                         help="CPU threads for torch (default: ~1/4 of the cores, "
                              "leaving headroom for the GPU task in progress)")
    parser.add_argument("--nice", type=int, default=15,
                         help="Lowered CPU priority (0-19, higher = more yielding). "
                              "0 to disable.")
    args = parser.parse_args()

    if args.nice:
        os.nice(args.nice)

    import torch
    torch.set_num_threads(args.threads)
    print(f"torch threads = {args.threads} (of {os.cpu_count()} available cores), "
          f"nice = {args.nice}, CUDA visible = {torch.cuda.is_available()}")

    # str.splitlines() wrongly breaks on real U+2028/U+2029 ("line
    # separator"/"paragraph separator") characters that sometimes appear
    # inside a table cell (found in one of the newly added rows, e.g.
    # PMC2874376: "MLST<U+2028>ST") -- a valid JSONL can only be split on
    # literal "\n" between records, hence the strict split here.
    rows = [json.loads(l) for l in INPUT_PATH.read_text().split("\n") if l.strip()]
    print(f"{len(rows)} candidate claims to normalize (full file)")

    # Same resumable key as 04_normalize_claims.py: (pmcid, sentence). Rows
    # already present (from the original document pool) are therefore
    # automatically skipped -- this only processes the new rows brought in
    # by the added documents, without retouching the earlier ones.
    done: set[tuple[str, str]] = set()
    if OUTPUT_PATH.exists():
        for line in OUTPUT_PATH.read_text().split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["pmcid"], r["sentence"]))
        print(f"Resuming: {len(done)} already normalized in {OUTPUT_PATH}")

    todo = [r for r in rows if (r["pmcid"], r["sentence"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} remaining to process (CPU)")

    if todo:
        agent = _CpuQwenAgent(MODEL_NAME, max_new_tokens=256)

    n_skipped = 0
    with OUTPUT_PATH.open("a") as f:
        for i, row in enumerate(todo):
            t0 = time.time()
            prompt = build_prompt(row)
            raw = agent.generate(prompt, greedy=True)
            answer = strip_thinking(raw).strip()
            answer = re.sub(r'^["\']|["\']$', "", answer).strip()
            dt = time.time() - t0

            if answer.upper() == "SKIP" or not answer:
                n_skipped += 1
                print(f"[{i+1}/{len(todo)}] {row['pmcid']} -> SKIP ({dt:.1f}s)")
                continue

            print(f"[{i+1}/{len(todo)}] {row['pmcid']} -> {answer[:100]} ({dt:.1f}s)")
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
