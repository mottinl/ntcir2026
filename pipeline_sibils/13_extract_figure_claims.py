#!/usr/bin/env python3
"""Step 2: finds sentences in body_sections that cite a figure (via its
label, e.g. "Figure 1"/"Fig. 1"), keeps the whole paragraph as context, then
filters "claim-worthy" sentences by heuristic -- same logic as
`03_extract_claims.py` for tables (regex/heuristic, no LLM/GPU at this
stage).

One output line = one (sentence, figure) candidate whose binary was
downloaded successfully (see `12_fetch_figures.py` / `figure_mentions.jsonl`),
NOT yet a Supported/Refuted pair -- figure perturbation comes next
(`14_perturb_figures.py`: Graph Swap / Category Swap / etc.).

Usage:
    python 13_extract_figure_claims.py
"""

import json
import re
from pathlib import Path

CACHE_DIR = Path("/data/pipeline_sibils_cache/raw")
FIGURE_MENTIONS_PATH = Path(__file__).parent / "figure_mentions.jsonl"
OUTPUT_PATH = Path(__file__).parent / "figure_claims.jsonl"

# Same shape as TABLE_MENTION_RE/TABLE_REF_SPAN_RE (03_extract_claims.py),
# "Fig(s)."/"Figure(s)" instead of "Table(s)" -- both abbreviated and spelled
# out forms appear across papers (SIBiLS echoes each paper's own convention
# verbatim, e.g. label="Fig. 1" in one PMCID, "Figure 1" in another, cf.
# figure_mentions.jsonl samples).
FIG_MENTION_RE = re.compile(r"\b(?:Figures?|Figs?\.?)\s+\d+\b", re.IGNORECASE)
FIG_REF_SPAN_RE = re.compile(
    r"\b(?:Figures?|Figs?\.?)\s+S?\d+(?:\s*(?:,|and|-|–|to)\s*S?\d+)*\b", re.IGNORECASE
)
EFIG_RE = re.compile(r"\be-?[Ff]igures?\b")
SUPP_FIG_RE = re.compile(
    r"\b(?:Figures?|Figs?\.?)\s+S\d+\b|\bsupplement(?:ary|al)?\s+(?:Figures?|Figs?\.?)\s+\d+\b",
    re.IGNORECASE,
)
CAPTION_ECHO_RE = re.compile(r"^(?:Figures?|Figs?\.?)\s*\d+\s*:", re.IGNORECASE)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
COMPARATIVE_RE = re.compile(
    r"\b(significant(ly)?|associat(ed|ion)|correlat(ed|ion)|compar(ed|ison)|"
    r"higher|lower|greater|less than|more than|increase(d)?|decrease(d)?|"
    r"outperform|versus|\bvs\.?\b|differ(ed|ence)?)\b",
    re.IGNORECASE,
)
DIGIT_RE = re.compile(r"\d")
LOW_CONTENT_RE = re.compile(r"\b(see|refer(s|red)?\s+to)\b", re.IGNORECASE)
MIN_WORDS = 8

FIGURE_LABEL_RE = re.compile(r"^(?:Figures?|Figs?\.?)\s*(\d+)", re.IGNORECASE)


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def is_claim_worthy(sentence: str) -> bool:
    n_words = len(sentence.split())
    if n_words < MIN_WORDS:
        return False
    sentence_wo_fig_ref = FIG_MENTION_RE.sub("", sentence)
    has_signal = bool(DIGIT_RE.search(sentence_wo_fig_ref)) or bool(COMPARATIVE_RE.search(sentence))
    if not has_signal:
        return False
    if LOW_CONTENT_RE.search(sentence) and not COMPARATIVE_RE.search(sentence):
        return False
    return True


def figures_by_label(doc: dict) -> dict[str, dict]:
    """Maps a figure's numeric label (e.g. "1") to its content block."""
    out = {}
    for section in doc.get("body_sections", []):
        for content in section.get("contents", []) or []:
            if content.get("tag") != "fig":
                continue
            m = FIGURE_LABEL_RE.match((content.get("label") or "").strip())
            if m:
                out[m.group(1)] = content
    return out


def load_downloaded_figures() -> dict[tuple[str, str], dict]:
    """(pmcid, xref_id) -> figure_mentions.jsonl record, restricted to
    figures whose binary was actually downloaded successfully (cf.
    12_fetch_figures.py -- 2/1843 had no matching S3 file)."""
    out = {}
    for line in FIGURE_MENTIONS_PATH.read_text().splitlines():
        r = json.loads(line)
        out[(r["pmcid"], r["xref_id"])] = r
    return out


def main():
    candidates = [json.loads(l) for l in
                  (Path(__file__).parent / "candidate_pmcids.jsonl").read_text().splitlines()]
    downloaded = load_downloaded_figures()

    n_docs = 0
    n_paragraphs_scanned = 0
    n_mentions_found = 0
    n_no_binary = 0
    n_claim_worthy = 0
    rows = []

    for c in candidates:
        pmcid = c["pmcid"]
        cache_path = CACHE_DIR / f"{pmcid}.json"
        if not cache_path.exists():
            continue
        doc = json.loads(cache_path.read_text())
        n_docs += 1

        figure_map = figures_by_label(doc)
        if not figure_map:
            continue

        for section in doc.get("body_sections", []):
            for content in section.get("contents", []) or []:
                if content.get("tag") != "p":
                    continue
                paragraph = content.get("text", "")
                if not FIG_MENTION_RE.search(paragraph):
                    continue
                n_paragraphs_scanned += 1

                for sentence in split_sentences(paragraph):
                    if not FIG_REF_SPAN_RE.search(sentence):
                        continue
                    if EFIG_RE.search(sentence) or SUPP_FIG_RE.search(sentence):
                        continue
                    if CAPTION_ECHO_RE.match(sentence.strip()):
                        continue

                    fig_numbers = set()
                    for span in FIG_REF_SPAN_RE.finditer(sentence):
                        fig_numbers.update(re.findall(r"\d+", span.group()))
                    if len(fig_numbers) != 1:
                        continue
                    fig_num = next(iter(fig_numbers))
                    figure = figure_map.get(fig_num)
                    if figure is None:
                        continue

                    xref_id = figure.get("xref_id")
                    binary = downloaded.get((pmcid, xref_id))
                    if binary is None:
                        n_no_binary += 1
                        continue

                    n_mentions_found += 1
                    if not is_claim_worthy(sentence):
                        continue
                    n_claim_worthy += 1

                    rows.append({
                        "pmcid": pmcid,
                        "bucket": c["bucket"],
                        "sentence": sentence,
                        "context": paragraph,
                        "figure_xref_id": xref_id,
                        "figure_label": figure.get("label"),
                        "figure_caption": figure.get("caption"),
                        "graphics": binary["graphics"],
                        "local_path": binary["local_path"],
                        "section_title": section.get("title"),
                    })

    with OUTPUT_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{n_docs} documents scanned")
    print(f"{n_paragraphs_scanned} paragraphs containing a figure mention")
    print(f"{n_mentions_found} sentences mentioning exactly one figure "
          f"(resolved + binary available, {n_no_binary} discarded for lack of a binary)")
    print(f"{n_claim_worthy} kept after the claim-worthy filter "
          f"({100 * n_claim_worthy / max(n_mentions_found, 1):.0f}%)")
    print(f"-> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
