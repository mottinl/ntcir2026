#!/usr/bin/env python3
"""Run Qwen3-VL-8B-Instruct over a SciClaimEval task2 split and produce
predictions in the format expected by evaluation_pipeline/eval/run_eval.py:

    [{"sample_id": "...", "pred_label": "evidence_id_1"|"evidence_id_2"}, ...]

Usage:
    python predict_task2.py --split dev --limit 5      # quick smoke test
    python predict_task2.py --split test                # full formal run
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

from data_utils import load_task2, resolve_path
from parsing import parse_task2_label
from qwen_agent import QwenVLAgent, strip_thinking, DEFAULT_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# prompt_task2.txt is plain Python source (not importable as-is since it's
# not a .py file); exec it to pull out the two templates so prompt_task2.txt
# stays the single place these prompts get edited/iterated on (mirrors
# predict_task1.py's prompt.txt convention).
_prompt_ns: dict = {}
exec(compile((Path(__file__).parent / "prompt_task2.txt").read_text(), "prompt_task2.txt", "exec"), _prompt_ns)
GENERIC_PROMPT_TEMPLATE = _prompt_ns["TASK2_GENERIC_PROMPT_TEMPLATE"]
FIGURE_PROMPT_TEMPLATE = _prompt_ns["TASK2_FIGURE_PROMPT_TEMPLATE"]

FALLBACK_LABEL = "evidence_id_1"


def build_prompt(record: dict) -> str:
    template = FIGURE_PROMPT_TEMPLATE if record.get("evi_type") == "figure" else GENERIC_PROMPT_TEMPLATE
    return template.format(
        claim=record["claim"],
        caption=record.get("caption", "N/A"),
        context=record.get("context", "N/A"),
        evi_type=record.get("evi_type", "evidence"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--output", default=None, help="Output predictions JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--load-in-4bit", action="store_true",
                         help="Load --model with on-the-fly bitsandbytes 4-bit quantization, split across 2 GPUs (for large non-AWQ checkpoints)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records (debugging)")
    parser.add_argument("--greedy", action="store_true", help="Greedy decoding instead of Qwen's recommended sampling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10, help="Write partial results every N records")
    parser.add_argument("--shard-index", type=int, default=0, help="Which shard this process handles (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards (run one process per GPU with CUDA_VISIBLE_DEVICES set)")
    parser.add_argument("--ids-file", default=None,
                         help="Only process sample_ids listed in this file (one per line, e.g. to "
                              "re-run just the records that fell back on a previous run). Sharding "
                              "(--num-shards/--shard-index) applies to this filtered list.")
    args = parser.parse_args()

    default_name = f"predictions_task2_{args.split}.json"
    if args.num_shards > 1:
        default_name = f"predictions_task2_{args.split}_shard{args.shard_index}of{args.num_shards}.json"
    output_path = Path(args.output or default_name)

    records = load_task2(args.split)
    if args.ids_file:
        wanted = {line.strip() for line in Path(args.ids_file).read_text().splitlines() if line.strip()}
        records = [r for r in records if r["sample_id"] in wanted]
        logger.info("--ids-file %s: restricted to %d/%d records", args.ids_file, len(records), len(wanted))
    if args.num_shards > 1:
        records = records[args.shard_index :: args.num_shards]
    if args.limit:
        records = records[: args.limit]

    done: dict[str, dict] = {}
    if output_path.exists():
        for row in json.loads(output_path.read_text()):
            done[row["sample_id"]] = row
        logger.info("Resuming: %d predictions already present in %s", len(done), output_path)

    import torch
    torch.manual_seed(args.seed)

    agent = QwenVLAgent(model_name=args.model, max_new_tokens=args.max_new_tokens, load_in_4bit=args.load_in_4bit)

    results = list(done.values())
    n_fallback = 0
    for i, record in enumerate(records):
        sample_id = record["sample_id"]
        if sample_id in done:
            continue

        image_path_1 = resolve_path(args.split, record["evidence_id_1"])
        image_path_2 = resolve_path(args.split, record["evidence_id_2"])
        prompt = build_prompt(record)

        try:
            image1 = Image.open(image_path_1).convert("RGB")
            image2 = Image.open(image_path_2).convert("RGB")
            raw = agent.generate([image1, image2], prompt, greedy=args.greedy)
            answer = strip_thinking(raw)
            label = parse_task2_label(answer)
            if label is None:
                logger.warning("[%s] could not parse label, defaulting to %s. Tail: %r",
                                sample_id, FALLBACK_LABEL, answer[-200:])
                label = FALLBACK_LABEL
                n_fallback += 1
        except Exception:
            logger.exception("[%s] generation failed, defaulting to %s", sample_id, FALLBACK_LABEL)
            label = FALLBACK_LABEL
            n_fallback += 1

        logger.info("[%d/%d] %s -> %s", i + 1, len(records), sample_id, label)
        results.append({"sample_id": sample_id, "pred_label": label})

        if (i + 1) % args.save_every == 0:
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Saved %d predictions to %s (%d fallback labels)", len(results), output_path, n_fallback)


if __name__ == "__main__":
    main()
