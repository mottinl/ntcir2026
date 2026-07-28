#!/usr/bin/env python3
"""Step 1: finds sentences in body_sections that cite a table (via its
label, e.g. "Table 1"), keeps the whole paragraph as context, then filters
"claim-worthy" sentences by heuristic (length, presence of digits/
comparatives) -- no LLM at this stage; a heuristic-first approach, with an
LLM as a fallback only if the false-positive rate turns out too high.

One output line = one (sentence, table) candidate, NOT yet a Supported/
Refuted pair -- table perturbation and PNG rendering come next
(05_perturb_table.py / 06_render_table_png.py).

Usage:
    python 03_extract_claims.py
"""

import json
import re
from pathlib import Path

CACHE_DIR = Path("/data/pipeline_sibils_cache/raw")
OUTPUT_PATH = Path(__file__).parent / "table_mentions.jsonl"

# "Table 1", "Table 1 and 2", "Tables 2-3" -- capture the label as it appears
# in-sentence, matched later against each table's own `label` field.
TABLE_MENTION_RE = re.compile(r"\bTables?\s+\d+\b", re.IGNORECASE)

# Full reference *span*, including comma/"and"-separated lists AND
# hyphen/en-dash ranges that don't repeat the word "Table" (e.g. "Tables 1,
# 2", "Tables 4 and 6", "Tables 2-4"/"Tables 2–4") -- used to detect
# multi-table sentences that TABLE_MENTION_RE alone would miss (it'd only see
# "Tables 1"/"Tables 4"/"Tables 2" and wrongly treat the sentence as
# single-table; the range case was found the same way as the S-label case
# below -- inspecting a normalized-claim sample turned up "Tables 2–4
# presents..." coming out tagged as single-table "Table 2"). The optional "S"
# also catches supplementary-table labels ("Table S1"), which a plain \d+
# silently ignores -- found when "Tables 1 and S3 ... Table S1" (three
# distinct tables) was passing as single-table because only the "1" in
# "Tables 1" registered as a digit reference.
TABLE_REF_SPAN_RE = re.compile(
    r"\bTables?\s+S?\d+(?:\s*(?:,|and|-|–|to)\s*S?\d+)*\b", re.IGNORECASE
)

# "eTable"/"e-Table" (supplementary material, e.g. JAMA-style) -- content not
# present in body_sections, see comment at its use site below.
ETABLE_RE = re.compile(r"\be-?[Tt]ables?\b")

# "Table S1" / "supplementary Table 1" -- supplementary tables, content not
# present in body_sections. Found a *silent misattribution* bug from this:
# "(Table S1)" and "(supplementary Table 1)" both extract digit "1" via
# TABLE_REF_SPAN_RE's optional "S" prefix, so they were resolving to the
# real main-text "Table 1" -- attaching a claim to the wrong table's data
# entirely (e.g. a sentence about supplementary PCR primers got paired with
# main Table 1's actual content, "Isolates of lyssavirus analysed in this
# study"). Must reject before the digit-extraction step, not just dedupe
# against it.
SUPP_TABLE_RE = re.compile(
    r"\bTables?\s+S\d+\b|\bsupplement(?:ary|al)?\s+Tables?\s+\d+\b", re.IGNORECASE
)

# "Table 1: Performance comparison ..." -- a caption/title being echoed as body
# text, not a real narrative sentence citing a table. Found in an "Additional
# files" listing section (PMC8855580) that recaps captions of *other*,
# separately-numbered supplementary tables (each additional file restarts its
# own "Table 1", "Table 2", ...) -- these got resolved against the real body
# Table 1 via its digit, silently misattributing 2 of 3 sentences to the
# wrong table (verified: real Table 1's caption is about TIS prediction, but
# one caught sentence was actually the *splice-site* table's own caption, the
# other the *poly(A) tail* table's). Reject at the source: no analytical claim
# is being made in a bare caption restatement anyway, so excluding is a
# no-loss fix, not just a bug patch.
CAPTION_ECHO_RE = re.compile(r"^Tables?\s*\d+\s*:", re.IGNORECASE)

# Basic sentence splitter: split after .?! followed by whitespace + a capital
# letter or digit. Not abbreviation-aware (e.g. "Fig. 2" mid-sentence can
# split early) -- acceptable for a heuristic-first pilot; revisit if the
# manual QC sample shows this fragmenting sentences badly.
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


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def is_claim_worthy(sentence: str) -> bool:
    n_words = len(sentence.split())
    if n_words < MIN_WORDS:
        return False
    # Strip the table self-reference ("Table 2") before checking for a digit
    # signal -- otherwise a purely descriptive sentence like "Table 2
    # presents a few examples..." passes just because of the "2" in "Table 2".
    sentence_wo_table_ref = TABLE_MENTION_RE.sub("", sentence)
    has_signal = bool(DIGIT_RE.search(sentence_wo_table_ref)) or bool(COMPARATIVE_RE.search(sentence))
    if not has_signal:
        return False
    # "See Table 2 for details" style non-claims: low-content phrasing AND no
    # comparative language (a low-content sentence that *also* compares
    # numbers, e.g. "As shown in Table 2, X was higher than Y", is kept).
    if LOW_CONTENT_RE.search(sentence) and not COMPARATIVE_RE.search(sentence):
        return False
    return True


TABLE_LABEL_RE = re.compile(r"^table\s*(\d+)", re.IGNORECASE)


def tables_by_label(doc: dict) -> dict[str, dict]:
    """Maps a table's numeric label (e.g. "1") to its content block.

    SIBiLS tags some non-table content ("Algorithm 1", "Listing 1") with
    tag=="table" too (found by inspecting a digit-key collision: a paper with
    both "Table 1" and "Algorithm 1" silently let "Algorithm 1" overwrite the
    real "Table 1" in this dict). Require the label to actually start with
    "Table" so those don't collide with real tables sharing the same number.
    """
    out = {}
    for section in doc.get("body_sections", []):
        for content in section.get("contents", []) or []:
            if content.get("tag") != "table":
                continue
            m = TABLE_LABEL_RE.match(content.get("label", "").strip())
            if m:
                out[m.group(1)] = content
    return out


def main():
    candidates = [json.loads(l) for l in
                  (Path(__file__).parent / "candidate_pmcids.jsonl").read_text().splitlines()]

    n_docs = 0
    n_paragraphs_scanned = 0
    n_mentions_found = 0
    n_claim_worthy = 0
    rows = []

    for c in candidates:
        pmcid = c["pmcid"]
        cache_path = CACHE_DIR / f"{pmcid}.json"
        if not cache_path.exists():
            continue
        doc = json.loads(cache_path.read_text())
        n_docs += 1

        table_map = tables_by_label(doc)
        if not table_map:
            continue

        for section in doc.get("body_sections", []):
            for content in section.get("contents", []) or []:
                if content.get("tag") != "p":
                    continue
                paragraph = content.get("text", "")
                if not TABLE_MENTION_RE.search(paragraph):
                    continue
                n_paragraphs_scanned += 1

                for sentence in split_sentences(paragraph):
                    if not TABLE_REF_SPAN_RE.search(sentence):
                        continue
                    # Multi-table sentences ("Table 1 and 2 show...",
                    # "Tables 4 and 6", "Tables 1, 2") are ambiguous re: which
                    # table's data grounds the claim -- skip for this pilot,
                    # single-table mentions only. Extract every digit inside
                    # each matched reference *span* (not just next to
                    # "Table") so comma/"and"-lists are correctly counted as
                    # multi-table.
                    # "eTable 1"/"Supplementary Table 1" style refs: content
                    # we don't have (not in body_sections), and \bTables?
                    # doesn't match the "eTable" compound word anyway, so a
                    # sentence that *also* cites a real "Table N" would
                    # otherwise slip through looking single-table even though
                    # part of its claim depends on content we can't see.
                    if ETABLE_RE.search(sentence) or SUPP_TABLE_RE.search(sentence):
                        continue
                    if CAPTION_ECHO_RE.match(sentence.strip()):
                        continue

                    table_numbers = set()
                    for span in TABLE_REF_SPAN_RE.finditer(sentence):
                        table_numbers.update(re.findall(r"\d+", span.group()))
                    if len(table_numbers) != 1:
                        continue
                    table_num = next(iter(table_numbers))
                    table = table_map.get(table_num)
                    if table is None:
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
                        "table_xref_id": table.get("xref_id"),
                        "table_label": table.get("label"),
                        "table_caption": table.get("caption"),
                        "table_columns": table.get("table_columns"),
                        "table_values": table.get("table_values"),
                        "section_title": section.get("title"),
                    })

    with OUTPUT_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{n_docs} documents scanned")
    print(f"{n_paragraphs_scanned} paragraphs containing a table mention")
    print(f"{n_mentions_found} sentences mentioning exactly one (resolved) table")
    print(f"{n_claim_worthy} kept after the claim-worthy filter "
          f"({100 * n_claim_worthy / max(n_mentions_found, 1):.0f}%)")
    print(f"-> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
