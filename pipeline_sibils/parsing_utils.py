"""Extract "Supported"/"Refuted" from a fine-tuned model's short reply.

Separate from pipeline_baseline/parsing.py because the training/eval prompt
here is the simple one from 08_finetune_qwen3vl8b.py, not pipeline_baseline's
Rec-3 prompt -- kept independent so editing one doesn't silently affect the
other.
"""

import re

_FINAL_LINE_RE = re.compile(r"final\s*answer.{0,30}?\b(supported|refuted)\b", re.IGNORECASE | re.DOTALL)
_ANY_LABEL_RE = re.compile(r"\b(supported|refuted)\b", re.IGNORECASE)


def parse_label(text: str) -> str | None:
    m = _FINAL_LINE_RE.search(text)
    if m:
        return m.group(1).capitalize()
    matches = _ANY_LABEL_RE.findall(text)
    if matches:
        return matches[-1].capitalize()
    return None
