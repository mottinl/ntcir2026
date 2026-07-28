#!/usr/bin/env python3
"""Pass A ("extraction"): run ONE advisor model over every unique image of
its scope (figure or table) referenced across dev+test task1+task2, caching
each reading to a resumable JSONL file. Never loaded in the same process as
the 32B decider (see README.md "Architecture") -- run this first, then predict_task1.py
/ predict_task2.py's --advisor-cache reads the finished cache.

Usage:
    python extract_advisor_cache.py --advisor chartgemma --limit 5   # smoke test
    python extract_advisor_cache.py --advisor chartgemma

    # paddleocr needs the ISOLATED venv (paddlepaddle-gpu conflicts with the
    # shared venv's torch build -- see advisor_paddleocr.py docstring):
    .venv-paddleocr/bin/python extract_advisor_cache.py --advisor paddleocr
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

from data_utils import load_task1, load_task2, resolve_path
from advisor_common import cache_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ADVISORS = {
    "chartgemma": ("advisor_chartgemma", "ChartGemmaAdvisor"),
    "paddleocr": ("advisor_paddleocr", "PaddleOCRAdvisor"),
}
# DePlot (google/deplot) was tried and dropped without a head-to-head
# comparison against ChartGemma -- see README.md "Pistes non retenues" for
# the rationale and how to bring it back if useful later.
# Table-LLaVA (table_llava) was replaced by PaddleOCR/PP-Structure after
# both a quantitative check (~7.5% of extractions were degenerate
# repetition-collapse) and qualitative spot-checks against real images
# (fabricated headers/row-labels, dropped columns) -- see README.md
# "Pistes non retenues". advisor_table_llava.py and convert_table_llava.py
# are kept on disk (not deleted) in case Table-LLaVA is revisited.


def evi_type_from_path(relative_path: str) -> str:
    """Derived from the path prefix rather than trusting a record's own
    evi_type field to apply identically to both task2 images -- verified
    consistent on current data (0 mismatches across 788 dev+test task2
    records) but not schema-enforced, so don't rely on the invariant."""
    if relative_path.startswith("tables_png/"):
        return "table"
    if relative_path.startswith("figures/"):
        return "figure"
    raise ValueError(f"Unrecognized evidence path prefix: {relative_path!r}")


def discover_images() -> list[tuple[str, str, str]]:
    """Returns deduped (split, relative_path, evi_type) tuples across
    dev+test task1+task2."""
    seen: dict[str, tuple[str, str, str]] = {}
    for split in ("dev", "test"):
        for record in load_task1(split):
            rel = record["evi_path"]
            key = cache_key(resolve_path(split, rel))
            seen.setdefault(key, (split, rel, evi_type_from_path(rel)))
        for record in load_task2(split):
            for field in ("evidence_id_1", "evidence_id_2"):
                rel = record[field]
                key = cache_key(resolve_path(split, rel))
                seen.setdefault(key, (split, rel, evi_type_from_path(rel)))
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advisor", required=True, choices=sorted(ADVISORS))
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images (debugging)")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    output_path = Path(args.output or f"advisor_cache_{args.advisor}.jsonl")

    module_name, class_name = ADVISORS[args.advisor]
    module = __import__(module_name)
    advisor_cls = getattr(module, class_name)
    scope = advisor_cls.SCOPE

    images = discover_images()
    images = [(split, rel, t) for split, rel, t in images if t == scope]
    logger.info("%d unique %s images to consider (advisor=%s)", len(images), scope, args.advisor)

    if args.num_shards > 1:
        images = images[args.shard_index :: args.num_shards]
    if args.limit:
        images = images[: args.limit]

    done: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text().split("\n"):
            if not line.strip():
                continue
            done.add(json.loads(line)["image_path"])
        logger.info("Resuming: %d entries already present in %s", len(done), output_path)

    todo = [(split, rel, t) for split, rel, t in images
            if cache_key(resolve_path(split, rel)) not in done]
    logger.info("%d images remaining to process", len(todo))

    if not todo:
        return

    advisor = advisor_cls()

    n_ok, n_fail = 0, 0
    with output_path.open("a") as f:
        for i, (split, rel, evi_type) in enumerate(todo):
            path = resolve_path(split, rel)
            key = cache_key(path)
            try:
                image = Image.open(path).convert("RGB")
                text = advisor.describe(image)
                row = {"image_path": key, "advisor": args.advisor, "evi_type": evi_type, "text": text}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += 1
            except Exception:
                logger.exception("[%s] advisor failed on %s, skipping (no cache entry written)", args.advisor, path)
                n_fail += 1
            logger.info("[%d/%d] %s", i + 1, len(todo), path)

    logger.info("Done: %d ok, %d failed, cache at %s", n_ok, n_fail, output_path)


if __name__ == "__main__":
    main()
