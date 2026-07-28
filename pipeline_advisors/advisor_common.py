"""Shared interface for this pipeline's chart/table "advisor" models.

Each advisor is a specialized model that reads ONE image and produces a
textual reading (a linearized data table, a detailed description, ...) which
gets cached to a JSONL file (one file per advisor, see
`extract_advisor_cache.py`) and later injected into the decider's prompt
(`predict_task1.py` / `predict_task2.py` via `--advisor-cache`). Advisors and
the 32B decider are never loaded in the same process -- see README.md
("Architecture") for why (GPU headroom: the 32B decider alone already uses
most of GPU0+GPU1's 20GB each).

Subclasses (advisor_chartgemma.py, advisor_paddleocr.py; advisor_table_llava.py
kept but no longer used, see README.md) implement `describe(image) -> str`
and set `NAME` / `SCOPE`.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Advisor:
    NAME: str = "base"
    SCOPE: str = "figure"  # "figure" or "table" -- which evi_type this advisor targets

    def describe(self, image) -> str:
        """image: PIL.Image (already opened, RGB). Returns the advisor's textual reading."""
        raise NotImplementedError


def cache_key(path) -> str:
    """Canonical cache key: resolved absolute path, so lookups from
    predict_task{1,2}.py (which build paths via data_utils.resolve_path) and
    extraction (which does the same) always agree regardless of cwd."""
    return str(Path(path).resolve())


def load_cache(path) -> dict[str, str]:
    """Reads an advisor cache JSONL into {image_path: text}. Returns {} if
    path is None or the file doesn't exist yet (caller treats a missing key
    as "no reading available", never fabricates one)."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("Advisor cache %s does not exist yet -- proceeding with no advisor text.", p)
        return {}
    cache: dict[str, str] = {}
    for line in p.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        cache[row["image_path"]] = row["text"]
    return cache
