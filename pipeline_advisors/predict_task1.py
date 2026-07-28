#!/usr/bin/env python3
"""Run Qwen3-VL-8B-Instruct over a SciClaimEval task1 split and produce
predictions in the format expected by evaluation_pipeline/eval/run_eval.py:

    [{"claim_id": "...", "pred_label": "Supported"|"Refuted"}, ...]

Usage:
    python predict_task1.py --split dev --limit 5      # quick smoke test
    python predict_task1.py --split test                # full formal run
    python predict_task1.py --split dev --advisor-cache advisor_cache_chartgemma.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

from data_utils import load_task1, resolve_path
from parsing import parse_task1_label
from qwen_agent import QwenVLAgent, strip_thinking, DEFAULT_MODEL
from advisor_common import cache_key, load_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# prompt.txt is plain Python source (not importable as-is since it's not a
# .py file); exec it to pull out REC3_USER_PROMPT_TEMPLATE so prompt.txt
# stays the single place the prompt itself gets edited/iterated on.
_prompt_ns: dict = {}
exec(compile((Path(__file__).parent / "prompt.txt").read_text(), "prompt.txt", "exec"), _prompt_ns)
PROMPT_TEMPLATE = _prompt_ns["REC3_USER_PROMPT_TEMPLATE"]

FALLBACK_LABEL = "Refuted"

ADVISOR_BLOCK_TEMPLATE = (
    "Automated chart/table reading (auxiliary, NOT ground truth -- may misread "
    "axes, swap rows/columns, or hallucinate values; treat it only as a hint "
    "and verify every number it states against the image yourself):\n{text}\n\n"
)


def build_prompt(record: dict, advisor_text: str | None = None) -> str:
    advisor_section = ADVISOR_BLOCK_TEMPLATE.format(text=advisor_text) if advisor_text else ""
    return PROMPT_TEMPLATE.format(
        claim=record["claim"],
        caption=record.get("caption", "N/A"),
        context=record.get("context", "N/A"),
        advisor_section=advisor_section,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--output", default=None, help="Output predictions JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--load-in-4bit", action="store_true",
                         help="Load --model with on-the-fly bitsandbytes 4-bit quantization, split across 2 GPUs (for large non-AWQ checkpoints)")
    parser.add_argument("--max-new-tokens", type=int, default=10240)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records (debugging)")
    parser.add_argument("--evi-type", default=None, choices=["table", "figure"],
                         help="Only process records of this evi_type (e.g. to compare figure-only advisors)")
    parser.add_argument("--greedy", action="store_true", help="Greedy decoding instead of Qwen's recommended sampling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10, help="Write partial results every N records")
    parser.add_argument("--shard-index", type=int, default=0, help="Which shard this process handles (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards (run one process per GPU with CUDA_VISIBLE_DEVICES set)")
    parser.add_argument("--advisor-cache", default=None,
                         help="Path to an advisor_cache_*.jsonl (extract_advisor_cache.py). "
                              "Omit for the pipeline_baseline-identical no-advisor baseline.")
    args = parser.parse_args()

    advisor_cache = load_cache(args.advisor_cache)

    default_name = f"predictions_task1_{args.split}.json"
    if args.num_shards > 1:
        default_name = f"predictions_task1_{args.split}_shard{args.shard_index}of{args.num_shards}.json"
    output_path = Path(args.output or default_name)

    records = load_task1(args.split)
    if args.evi_type:
        records = [r for r in records if r.get("evi_type") == args.evi_type]
    if args.num_shards > 1:
        records = records[args.shard_index :: args.num_shards]
    if args.limit:
        records = records[: args.limit]

    done: dict[str, dict] = {}
    if output_path.exists():
        for row in json.loads(output_path.read_text()):
            done[row["claim_id"]] = row
        logger.info("Resuming: %d predictions already present in %s", len(done), output_path)

    import torch
    torch.manual_seed(args.seed)

    agent = QwenVLAgent(model_name=args.model, max_new_tokens=args.max_new_tokens, load_in_4bit=args.load_in_4bit)

    results = list(done.values())
    n_fallback = 0
    for i, record in enumerate(records):
        claim_id = record["claim_id"]
        if claim_id in done:
            continue

        image_path = resolve_path(args.split, record["evi_path"])
        advisor_text = advisor_cache.get(cache_key(image_path))
        prompt = build_prompt(record, advisor_text)

        try:
            image = Image.open(image_path).convert("RGB")
            raw = agent.generate([image], prompt, greedy=args.greedy, max_new_tokens=args.max_new_tokens)
            answer = strip_thinking(raw)
            label = parse_task1_label(answer)
            if label is None:
                logger.warning("[%s] could not parse label, defaulting to %s. Tail: %r",
                                claim_id, FALLBACK_LABEL, answer[-200:])
                label = FALLBACK_LABEL
                n_fallback += 1
        except Exception:
            logger.exception("[%s] generation failed, defaulting to %s", claim_id, FALLBACK_LABEL)
            label = FALLBACK_LABEL
            n_fallback += 1

        logger.info("[%d/%d] %s -> %s", i + 1, len(records), claim_id, label)
        results.append({"claim_id": claim_id, "pred_label": label})

        if (i + 1) % args.save_every == 0:
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Saved %d predictions to %s (%d fallback labels)", len(results), output_path, n_fallback)


if __name__ == "__main__":
    main()
