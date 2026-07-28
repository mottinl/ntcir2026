#!/usr/bin/env python3
"""Task2 evidence selection via task1 decomposition.

task2's two candidates are, per the dataset's construction, the genuine
evidence and a perturbed distractor for the same claim (see the "operation"
field on dev records, e.g. "Swap rows or columns") -- exactly the
original-vs-perturbed relationship task1's Supported/Refuted label encodes
for a single image. So: classify each candidate independently with task1's
prompt; if exactly one comes back Supported, pick it (decisive). Only fall
back to the joint two-image task2 prompt (predict_task2.py's approach) when
both candidates get the same verdict (tie).

Estimated from task1's dev confusion matrix (recall Supported 75.7%, recall
Refuted 85.8%, assuming the two independent calls' errors are uncorrelated):
~65% decisive-and-correct, ~3.5% decisive-and-wrong, ~31.6% falling back to
the joint prompt (accuracy in line with its current ~67.6% score) -> a
projected ~86% overall accuracy, vs. 67.61% for the joint prompt alone. The
independence assumption is optimistic (same claim/context feeds both calls,
so their errors are probably somewhat correlated), but the projected gap
leaves comfortable margin even if reality falls short of the estimate.

The per-record decision logic (decide_pair) takes an `agent`-shaped object
(anything with .generate(images, text, greedy=...) -> str) so it can be
exercised with a stub in test_task2_decomposed.py without loading any model
-- see that file for a CPU-only, no-GPU smoke test of the branching logic
before spending real GPU time on a dev-split run. This branching logic was
validated that way but never run end-to-end on GPU, so it wasn't used for
the submitted predictions -- see ../README.md for what was.

Usage:
    python predict_task2_decomposed.py --split dev --limit 5      # smoke test
    python predict_task2_decomposed.py --split dev                # full dev eval
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

from data_utils import load_task2, resolve_path
from parsing import parse_task1_label, parse_task2_label
from qwen_agent import QwenVLAgent, strip_thinking, DEFAULT_MODEL
import predict_task1
import predict_task2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FALLBACK_LABEL = "evidence_id_1"


def decide_pair(agent, record: dict, image1, image2, greedy: bool = False,
                 max_new_tokens_single: int | None = None,
                 max_new_tokens_joint: int | None = None) -> tuple[str, dict]:
    """Returns (pred_label, meta). meta always has 'mode' ('decisive' or
    'fallback') and 'verdicts' (the two independent task1 verdicts), so a
    dev run can report how often each path fires and validate the decisive
    branch's accuracy separately from the fallback branch's.
    """
    single_prompt = predict_task1.build_prompt(record)

    raw1 = agent.generate([image1], single_prompt, greedy=greedy, max_new_tokens=max_new_tokens_single)
    raw2 = agent.generate([image2], single_prompt, greedy=greedy, max_new_tokens=max_new_tokens_single)
    verdict1 = parse_task1_label(strip_thinking(raw1))
    verdict2 = parse_task1_label(strip_thinking(raw2))

    if verdict1 == "Supported" and verdict2 != "Supported":
        return "evidence_id_1", {"mode": "decisive", "verdicts": [verdict1, verdict2]}
    if verdict2 == "Supported" and verdict1 != "Supported":
        return "evidence_id_2", {"mode": "decisive", "verdicts": [verdict1, verdict2]}

    joint_prompt = predict_task2.build_prompt(record)
    raw_joint = agent.generate([image1, image2], joint_prompt, greedy=greedy, max_new_tokens=max_new_tokens_joint)
    label = parse_task2_label(strip_thinking(raw_joint)) or FALLBACK_LABEL
    return label, {"mode": "fallback", "verdicts": [verdict1, verdict2]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--output", default=None, help="Output predictions JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens-single", type=int, default=10240,
                         help="task1-style per-image verdict call (default matches predict_task1.py)")
    parser.add_argument("--max-new-tokens-joint", type=int, default=2048,
                         help="task2-style joint fallback call (default matches predict_task2.py)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    default_name = f"predictions_task2_decomposed_{args.split}.json"
    if args.num_shards > 1:
        default_name = f"predictions_task2_decomposed_{args.split}_shard{args.shard_index}of{args.num_shards}.json"
    output_path = Path(args.output or default_name)
    debug_path = output_path.with_name(output_path.stem + "_debug.json")

    records = load_task2(args.split)
    if args.num_shards > 1:
        records = records[args.shard_index :: args.num_shards]
    if args.limit:
        records = records[: args.limit]

    done: dict[str, dict] = {}
    if output_path.exists():
        for row in json.loads(output_path.read_text()):
            done[row["sample_id"]] = row
        logger.info("Resuming: %d predictions already present in %s", len(done), output_path)
    debug_rows: list[dict] = json.loads(debug_path.read_text()) if debug_path.exists() else []

    import torch
    torch.manual_seed(args.seed)

    agent = QwenVLAgent(model_name=args.model, load_in_4bit=args.load_in_4bit)

    results = list(done.values())
    n_decisive = sum(1 for r in debug_rows if r["mode"] == "decisive")
    n_fallback = sum(1 for r in debug_rows if r["mode"] == "fallback")
    for i, record in enumerate(records):
        sample_id = record["sample_id"]
        if sample_id in done:
            continue

        try:
            image1 = Image.open(resolve_path(args.split, record["evidence_id_1"])).convert("RGB")
            image2 = Image.open(resolve_path(args.split, record["evidence_id_2"])).convert("RGB")
            label, meta = decide_pair(
                agent, record, image1, image2, greedy=args.greedy,
                max_new_tokens_single=args.max_new_tokens_single,
                max_new_tokens_joint=args.max_new_tokens_joint,
            )
            if meta["mode"] == "decisive":
                n_decisive += 1
            else:
                n_fallback += 1
        except Exception:
            logger.exception("[%s] generation failed, defaulting to %s", sample_id, FALLBACK_LABEL)
            label, meta = FALLBACK_LABEL, {"mode": "error", "verdicts": [None, None]}

        logger.info("[%d/%d] %s -> %s (%s, verdicts=%s)",
                     i + 1, len(records), sample_id, label, meta["mode"], meta["verdicts"])
        results.append({"sample_id": sample_id, "pred_label": label})
        debug_rows.append({"sample_id": sample_id, "pred_label": label, **meta})

        if (i + 1) % args.save_every == 0:
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            debug_path.write_text(json.dumps(debug_rows, indent=2, ensure_ascii=False))

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    debug_path.write_text(json.dumps(debug_rows, indent=2, ensure_ascii=False))
    logger.info("Saved %d predictions to %s (%d decisive, %d fallback)",
                len(results), output_path, n_decisive, n_fallback)


if __name__ == "__main__":
    main()
