"""Path/loading helpers for the evaluation_pipeline data layout.

evaluation_pipeline/data/<split>/data/ holds the release JSONs plus
figures/, tables_png/, tables/, papers/ -- see evaluation_pipeline/README.md.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DATA_ROOT = REPO_ROOT / "evaluation_pipeline" / "data"


def data_root_for(split: str) -> Path:
    return EVAL_DATA_ROOT / split / "data"


def load_task1(split: str) -> list[dict]:
    path = data_root_for(split) / f"{split}_task1_release.json"
    return json.loads(path.read_text())


def load_task2(split: str) -> list[dict]:
    path = data_root_for(split) / f"{split}_task2_release.json"
    return json.loads(path.read_text())


def resolve_path(split: str, relative_path: str) -> Path:
    return data_root_for(split) / relative_path
