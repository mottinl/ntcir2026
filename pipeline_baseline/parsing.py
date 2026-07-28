"""Extract the final structured label from a Qwen3-VL response."""

import re

# Matches both the plain "Final answer: Supported" ending and the REC2
# prompt's "...the final answer is: Answer: Supported" ending (prompt.txt).
_FINAL_LINE_T1 = re.compile(r"final\s*answer.{0,30}?\b(supported|refuted)\b", re.IGNORECASE | re.DOTALL)
_ANY_LABEL_T1 = re.compile(r"\b(supported|refuted)\b", re.IGNORECASE)

# Two response formats seen in the wild: predict_task2.py's original prompt
# asked for "Final answer: evidence_id_N" directly; prompt_task2.txt's
# generic/figure templates instead end with "...final answer is: Answer: $N"
# (a bare 1 or 2). Match both, normalizing to "evidence_id_N".
_FINAL_LINE_T2 = re.compile(
    r"final\s*answer.{0,40}?(?:(evidence_id_[12])|\banswer\s*:?\s*\$?\s*([12])\b)",
    re.IGNORECASE | re.DOTALL,
)
_ANY_LABEL_T2 = re.compile(r"\b(evidence_id_[12])\b", re.IGNORECASE)
_ANY_ANSWER_T2 = re.compile(r"\banswer\s*:?\s*\$?\s*([12])\b", re.IGNORECASE)


def parse_task1_label(text: str) -> str | None:
    m = _FINAL_LINE_T1.search(text)
    if m:
        return m.group(1).capitalize()
    matches = _ANY_LABEL_T1.findall(text)
    if matches:
        return matches[-1].capitalize()
    return None


def parse_task2_label(text: str) -> str | None:
    m = _FINAL_LINE_T2.search(text)
    if m:
        evidence_id, digit = m.group(1), m.group(2)
        return evidence_id.lower() if evidence_id else f"evidence_id_{digit}"
    matches = _ANY_LABEL_T2.findall(text)
    if matches:
        return matches[-1].lower()
    matches = _ANY_ANSWER_T2.findall(text)
    if matches:
        return f"evidence_id_{matches[-1]}"
    return None
