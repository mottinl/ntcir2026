#!/usr/bin/env python3
"""Step 1: perturbs the evidence (table_values) to generate the Refuted
counterpart of each claim -- same claim, modified table (mirroring the
official dataset's own construction: the claim doesn't change, the evidence
is what gets perturbed).

Default operation: **"Change the cell values" only**. Verified by diffing
several real Supported/Refuted pairs from dev_task1/2: the `peerj`
sub-domain (our target) uses this operation for 100% of its table pairs
(0/100 "Swap rows or columns", "Category Swap" doesn't appear at all, even
outside peerj) -- so there's no need to replicate those to match the dev
set's real distribution.

`--swap-fraction` (v2) adds a second operation, **"Swap rows or columns"**,
for a fraction of the examples (a deterministic choice via an rng seeded by
(pmcid, sentence), so reproducible) -- not to better match the dev set's
real distribution (which doesn't use it for peerj, see above), but to
diversify the training signal. Motivation: v1 (single operation) reaches a
near-zero train_loss but an internal validation accuracy close to chance
(52%) and a clear regression on the real dev set -- a sign the model
memorizes the signature of ONE operation rather than learning to verify a
table. Swaps one table row against another (data columns only, the row
label in column 0 stays put) -- a structurally different kind of corruption
from a single modified cell value, forcing a comparison across several
rows rather than spotting one isolated number.

Targeting strategy (inferred by diffing real pairs, e.g. claim_id_pair
0263: "having children (p = 0.002)" -> only the p-value cell changed to
0.200, nothing else modified):
1. **"Claim-grounded" cell**: looks in table_values for a cell whose number
   also appears in the claim -- directly targets the figure the claim
   asserts, for a Refuted contrast that makes sense (a reader who checks
   that specific figure against the table finds it wrong). Generic numbers
   (integers <10, self-references like "Table N"/"Figure N") are excluded
   from matching to avoid a false-positive match on an insignificant number.
2. **Random fallback**: if no number from the claim matches a cell (table
   too coarse, claim only qualitative), perturbs a randomly chosen numeric
   cell (deterministic seed per (pmcid, sentence)).

Value modification:
- A cell that looks like a pure p-value (the whole cell content is just
  "0.002" or "<0.001", nothing else) -> pushed across the 0.05 significance
  threshold, as observed in several real pairs (0.002->0.200,
  <0.001<->0.202, 0.255-><0.001).
- Otherwise -> a substantial multiplicative variation (factor outside
  [0.85, 1.15]) to stay in a plausible format while being clearly different.

Accepted simplification (pilot): only one cell modified per table, no
propagation of arithmetic consistency to related cells (e.g. a percentage
in parentheses next to a modified count) -- the official dataset does this
sometimes, but replicating it exactly would require understanding each
column's semantics, out of scope for a pure function in a first training set.

v3: the task organizers published the official table-edit taxonomy: "Value
changes, row/column swaps, full table altercations, value scaling, caption
adjustments" -- only 2/5 were implemented so far. Three new operations added
to better match this taxonomy (selected via `--scale-fraction`/
`--caption-fraction`, same fallback logic as `--swap-fraction` -- if the
chosen operation finds no suitable target, falls back to "Change the cell
values"):
- **"Value scaling"**: multiplies ALL numeric values in one column by a
  common round factor (x10/x100/÷10/÷100/x2/÷2/x5/÷5, a unit-error style
  mistake) -- structurally different from a single cell edit: the whole
  column shifts consistently, not just one number.
- **"Caption adjustments"** (rule-based, not an LLM rewrite -- partial
  coverage accepted rather than a new stylistic-leakage risk): the TABLE
  stays unchanged (same rendered pixels for Supported and Refuted), only
  the caption (`table_caption`) is modified -- looks for a specific number
  in the caption (excluding self-references like "Table N"/"Figure N", same
  filter as for the claim) and perturbs it with the same mechanism as
  `perturb_value`. If the caption has no verifiable number, the operation
  isn't applicable for that row (falls back to "Change the cell values") --
  coverage is, by construction, limited to captions that state a numeric fact.

**"Full table altercations" (2-4 independent cell edits) was implemented
and validated, then removed (user decision)**: it's the heaviest of the 5
operations in visual-token load (mean post-resize area 1.25 Mpx vs.
1.09-1.14 for the others) and the likely identified cause of the v4
fine-tuning run's persistent CUDA OOM on this virtualized GPU (never
completed in 3 attempts). Too risky to carry into a new attempt for a
diversity operation that peerj doesn't even use in its real pairs anyway
(0% observed, see above) -- reintroduce only behind a real fix for the OOM.

Usage:
    python 05_perturb_table.py [--limit N]
"""

import argparse
import json
import random
import re
from pathlib import Path

INPUT_PATH = Path(__file__).parent / "claims_normalized.jsonl"
OUTPUT_PATH = Path(__file__).parent / "perturbed_claims.jsonl"

# Decimal part requires >=1 digit after the dot -- matching a bare trailing
# "." (e.g. "...taken from study 1." at end of sentence) as a decimal point
# was a real bug: it made a generic "1" look "specific" (see _is_specific)
# and produced no-op perturbations (perturbed value rounds back to the same
# integer).
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
TABLE_FIG_REF_RE = re.compile(r"\b(?:Tables?|Figs?\.?|Figures?)\s+S?\d+\b", re.IGNORECASE)

# Citation cells ("Smith et al. (2018)", "Smith et al [35]", "Ng and Tan
# (2024)") must never be treated as data candidates: found via QC sampling
# that these are extremely common in comparison/review tables, and a
# citation's year or reference number very often numerically coincides with
# some *unrelated* number in the claim (frequently the claim's *own* inline
# citation, to a different paper) -- e.g. claim citing "(Thomas et al.,
# 2020)" spuriously grounding a match onto an unrelated "Sherkatghanad et
# al. (2020)" table cell. Not a data value at all, so no amount of
# specificity tuning fixes this -- the cell itself must be excluded from
# candidacy. Two-author citations ("Ng and Tan (2024)", no "et al") slipped
# through an earlier version that only checked for "et al" -- broadened to
# the general "Capitalized name(s) (year)" shape.
CITATION_CELL_RE = re.compile(
    r"(?i:\bet al\b)"
    r"|[A-Z][a-zA-Z'\-]+(?:\s+(?:and|&)\s+[A-Z][a-zA-Z'\-]+)?\s*[\(\[]\d{4}[\)\]]"
)

# Bare 4-digit numbers in a plausible publication-year range are excluded
# from claim targets for the same reason -- almost always an inline
# citation year in the claim text, not a data value to ground on.
def _is_citation_year(value: float, raw: str) -> bool:
    return "." not in raw and 1900 <= value <= 2099


# Only numbers that look "specific" enough are used to ground a perturbation
# to the claim -- a bare single digit (a count, a list index, ...) is too
# likely to spuriously match an unrelated cell.
def _is_specific(value: float, raw: str) -> bool:
    return ("." in raw or abs(value) >= 10) and not _is_citation_year(value, raw)


def _parse_num(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _is_identifier_digit(cell: str, span: tuple[int, int]) -> bool:
    """True if the matched number is glued to a letter (e.g. the "1" in gene/
    protein identifiers like "MDH1", "SED1", "1CFD", "HMOX1"). Biomedical
    result tables are full of these -- perturbing the digit would either be a
    silent no-op or, worse, mint a fake identifier that doesn't exist, found
    via QC sampling ("MDH1", "HMOX1" picked as fallback perturbation
    targets)."""
    start, end = span
    before = cell[start - 1] if start > 0 else ""
    after = cell[end] if end < len(cell) else ""
    return before.isalpha() or after.isalpha()


def extract_numbers(text: str) -> list[tuple[float, str]]:
    """[(value, raw_string), ...] for every number in text."""
    out = []
    for m in NUM_RE.finditer(text):
        v = _parse_num(m.group())
        if v is not None:
            out.append((v, m.group()))
    return out


def claim_target_numbers(claim: str) -> set[float]:
    stripped = TABLE_FIG_REF_RE.sub("", claim)
    return {v for v, raw in extract_numbers(stripped) if _is_specific(v, raw)}


def find_claim_grounded_cell(table_values: list[list[str]], targets: set[float]):
    """First (row, col, match_span, matched_value) whose cell contains a
    number equal (within tolerance) to one of the claim's target numbers."""
    for i, row in enumerate(table_values):
        for j, cell in enumerate(row):
            if not cell or CITATION_CELL_RE.search(cell):
                continue
            for m in NUM_RE.finditer(cell):
                if _is_identifier_digit(cell, m.span()):
                    continue
                v = _parse_num(m.group())
                if v is None:
                    continue
                if any(abs(v - t) < 1e-6 for t in targets):
                    return i, j, m.span(), v
    return None


def find_random_numeric_cell(table_values: list[list[str]], rng: random.Random):
    candidates = []
    for i, row in enumerate(table_values):
        for j, cell in enumerate(row):
            if not cell or CITATION_CELL_RE.search(cell):
                continue
            matches = [m for m in NUM_RE.finditer(cell) if not _is_identifier_digit(cell, m.span())]
            if matches:
                candidates.append((i, j, matches))
    if not candidates:
        return None
    i, j, matches = rng.choice(candidates)
    m = rng.choice(matches)
    v = _parse_num(m.group())
    return i, j, m.span(), v


def is_pure_pvalue_cell(cell: str) -> bool:
    """Whole cell is just a (possibly '<'-prefixed) decimal < 1 -- no other
    content (percentages/CIs etc. always carry brackets/units alongside)."""
    return bool(re.match(r"^\s*<?\s*0?\.\d+\s*$", cell))


def _format(new_value: float, decimals: int) -> str:
    if decimals == 0:
        return str(int(round(new_value)))
    return f"{new_value:.{decimals}f}"


def perturb_value(value: float, raw: str, cell: str, rng: random.Random) -> str:
    """Guaranteed to return something != raw -- rounding a small jittered
    value back to the same displayed digits (e.g. 1 * 1.3 = 1.3 -> "1", or
    0 * any factor = 0) was a real bug found via QC, silently producing
    no-op "perturbations". Retries with progressively more aggressive jitter
    rather than trusting one random draw to land on a different value."""
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    is_percent = "%" in cell
    is_pvalue = is_pure_pvalue_cell(cell)

    for attempt in range(12):
        if is_pvalue:
            if value < 0.05:
                new_value = round(rng.uniform(0.06, 0.95), max(decimals, 2))
            else:
                new_value = round(rng.uniform(0.0005, 0.049), max(decimals, 3))
            if new_value < 0.001:
                result = "<0.001"
            else:
                result = f"{new_value:.{max(decimals, 2)}f}"
            if result != raw and result != cell.strip():
                return result
            continue

        if decimals == 0 and 1900 <= value <= 2035 and not is_percent:
            # A bare year (e.g. "2015-2020" search-period cell) perturbed by
            # the generic multiplicative jitter can land on an absurd value
            # like 2832 -- found via QC. Stay within a plausible year range
            # instead, localized around the original.
            lo, hi = max(1900, int(value) - 15), min(2035, int(value) + 15)
            choices = [y for y in range(lo, hi + 1) if y != int(value)]
            new_value = float(rng.choice(choices))
        elif value == 0:
            # Multiplicative jitter can't move zero at all -- use an
            # additive draw scaled to a plausible magnitude instead, growing
            # with each retry in case decimals round it back to 0.
            scale = (1 + attempt) * (10 ** (-decimals) if decimals else 1)
            new_value = round(rng.uniform(scale, scale * 3), decimals)
        else:
            spread = 0.3 + 0.1 * attempt  # widen the jitter range on retry
            factor = rng.uniform(1 + spread, 1 + spread * 1.4) if rng.random() < 0.5 \
                else rng.uniform(1 - spread * 1.4, 1 - spread)
            factor = max(factor, 0.01)
            new_value = value * factor
            if value >= 0:
                new_value = max(new_value, 0.0)

        # A percentage cell reading e.g. "104%" is implausible on its face
        # (no need to even check the claim to spot it) -- clamp to a valid
        # range while still forcing a real change from the original.
        if is_percent:
            new_value = min(new_value, 100.0)
            if abs(new_value - value) < 1:
                new_value = 0.0 if value > 50 else 100.0

        result = _format(new_value, decimals)
        if result != raw:
            return result

    # Exhausted retries (pathological case, e.g. decimals=0 and value so
    # small every rounded draw collides) -- guaranteed-different fallback.
    return _format(value + (1 if value >= 0 else -1) + (0 if decimals == 0 else 0.5), decimals)


def perturb_cell(cell: str, span: tuple[int, int], new_number: str) -> str:
    start, end = span
    return cell[:start] + new_number + cell[end:]


def find_swappable_row_pair(table_values: list[list[str]], rng: random.Random, claim: str):
    """Group data rows (>=2 non-empty cells, so section-header/blank rows are
    excluded) by length -- only rows of equal length can be swapped cleanly.
    Prefers a row whose label (column 0) is referenced in the claim, so the
    swap lands on data the claim actually talks about, same grounded/fallback
    spirit as the cell-value operation above."""
    from collections import defaultdict
    by_length = defaultdict(list)
    for i, row in enumerate(table_values):
        non_empty = [c for c in row if c and c.strip()]
        if len(row) >= 2 and len(non_empty) >= 2:
            by_length[len(row)].append(i)

    claim_lower = claim.lower()
    for length, idxs in by_length.items():
        if len(idxs) < 2:
            continue
        for i in idxs:
            label = (table_values[i][0] or "").strip()
            if len(label) >= 4 and label.lower() in claim_lower:
                j = rng.choice([x for x in idxs if x != i])
                return i, j, True

    eligible = [length for length, idxs in by_length.items() if len(idxs) >= 2]
    if not eligible:
        return None
    length = rng.choice(eligible)
    i, j = rng.sample(by_length[length], 2)
    return i, j, False


def perform_row_swap(table_values: list[list[str]], rng: random.Random, claim: str):
    """Swaps the data columns (everything but column 0, the row label) of two
    rows. Retries with a different pair if the draw happens to pick two rows
    whose data is already identical (a real no-op, same class of bug as the
    cell-value perturber's guaranteed-different retry loop)."""
    for _ in range(5):
        hit = find_swappable_row_pair(table_values, rng, claim)
        if hit is None:
            return None
        i, j, grounded = hit
        if table_values[i][1:] == table_values[j][1:]:
            continue
        perturbed = [r.copy() for r in table_values]
        perturbed[i][1:], perturbed[j][1:] = table_values[j][1:], table_values[i][1:]
        return perturbed, i, j, grounded
    return None


SCALE_FACTORS = [10, 100, 0.1, 0.01, 2, 0.5, 5, 0.2]


def find_scalable_column(table_values: list[list[str]], claim_targets: set[float]):
    """Groups numeric cells by column (column 0 excluded -- it's the row
    label, not data). Returns (col, [(row, span, value), ...], grounded) for
    a column referencing a claim target number if one exists, else the
    first column with >=1 numeric cell, or None if the table has no numeric
    data column at all."""
    from collections import defaultdict
    col_cells = defaultdict(list)
    for i, row in enumerate(table_values):
        for j, cell in enumerate(row):
            if j == 0 or not cell or CITATION_CELL_RE.search(cell):
                continue
            for m in NUM_RE.finditer(cell):
                if _is_identifier_digit(cell, m.span()):
                    continue
                v = _parse_num(m.group())
                if v is not None:
                    col_cells[j].append((i, m.span(), v))

    if not col_cells:
        return None

    for j, cells in col_cells.items():
        if any(abs(v - t) < 1e-6 for _, _, v in cells for t in claim_targets):
            return j, cells, True
    j = next(iter(col_cells))
    return j, col_cells[j], False


def perform_value_scaling(table_values: list[list[str]], rng: random.Random, claim: str):
    """Scales every numeric cell in one column by a common round factor
    (x10/÷10/... -- a unit-error style mistake), instead of a single cell's
    value -- a whole column shifted consistently is a structurally
    different corruption from one isolated wrong number."""
    targets = claim_target_numbers(claim)
    hit = find_scalable_column(table_values, targets)
    if hit is None:
        return None
    col, cells, grounded = hit

    factor = rng.choice(SCALE_FACTORS)
    perturbed = [r.copy() for r in table_values]
    changed_rows = []
    for i, span, v in cells:
        raw = table_values[i][col][span[0]:span[1]]
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        new_str = _format(v * factor, decimals)
        if new_str == raw:
            continue  # e.g. 0 * factor -- no visible change for this cell, skip it
        perturbed[i][col] = perturb_cell(table_values[i][col], span, new_str)
        changed_rows.append(i)

    if not changed_rows:
        return None
    return perturbed, col, factor, grounded, changed_rows


def _all_numeric_cells(table_values: list[list[str]]):
    candidates = []
    for i, row in enumerate(table_values):
        for j, cell in enumerate(row):
            if not cell or CITATION_CELL_RE.search(cell):
                continue
            matches = [m for m in NUM_RE.finditer(cell) if not _is_identifier_digit(cell, m.span())]
            if matches:
                candidates.append((i, j, matches))
    return candidates



def find_caption_number(caption: str, claim: str):
    """Returns (span, raw, value, grounded) for a specific, perturbable
    number in the caption (excluding "Table N"/"Figure N" self-references,
    citation years, identifier-glued digits like "SF-36" (NUM_RE's leading
    `-?` consumes the hyphen as a sign, so `_is_identifier_digit` -- written
    for table cells -- also correctly catches this caption case: the
    character right before the match is the letter preceding the hyphen),
    and non-specific digits -- same filters as the claim grounding above),
    preferring one also referenced in the claim. None if the caption states
    nothing numerically checkable -- by design, this caps how many rows
    "Caption adjustments" can apply to rather than forcing an edit onto
    prose that isn't a verifiable claim."""
    caption = caption or ""
    excluded_spans = [m.span() for m in TABLE_FIG_REF_RE.finditer(caption)]

    def _excluded(span):
        return any(a <= span[0] < b for a, b in excluded_spans)

    claim_targets = claim_target_numbers(claim)
    matches = []
    for m in NUM_RE.finditer(caption):
        if _excluded(m.span()) or _is_identifier_digit(caption, m.span()):
            continue
        v = _parse_num(m.group())
        if v is None or not _is_specific(v, m.group()) or _is_citation_year(v, m.group()):
            continue
        matches.append((m.span(), m.group(), v))

    if not matches:
        return None
    for span, raw, v in matches:
        if any(abs(v - t) < 1e-6 for t in claim_targets):
            return span, raw, v, True
    span, raw, v = matches[0]
    return span, raw, v, False


def perform_caption_adjustment(caption: str, rng: random.Random, claim: str):
    """Table stays byte-identical (perturbed_table_values == table_values --
    Supported and Refuted render to the same image); only the caption text
    is perturbed, so the contradiction lives in the legend, not the table
    pixels -- a mechanism our renderer/table-perturbation ops can't produce
    at all."""
    hit = find_caption_number(caption, claim)
    if hit is None:
        return None
    span, raw, value, grounded = hit
    new_number = perturb_value(value, raw, caption, rng)
    perturbed_caption = caption[:span[0]] + new_number + caption[span[1]:]
    return perturbed_caption, grounded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--swap-fraction", type=float, default=0.0,
                         help="Fraction of examples using 'Swap rows or columns' instead of "
                              "'Change the cell values' (deterministic per-row draw).")
    parser.add_argument("--scale-fraction", type=float, default=0.0,
                         help="Fraction of examples using 'Value scaling' (whole-column x10/÷10/... factor).")
    parser.add_argument("--caption-fraction", type=float, default=0.0,
                         help="Fraction of examples using 'Caption adjustments' (table image unchanged, "
                              "legend perturbed instead -- only applies where the caption states a "
                              "specific checkable number, falls back otherwise).")
    args = parser.parse_args()

    # str.splitlines() wrongly breaks on real U+2028/U+2029 characters
    # found inside a table cell (e.g. PMC2874376: "MLST<U+2028>ST") -- strict
    # split here (same fix as in 04_normalize_claims.py).
    rows = [json.loads(l) for l in INPUT_PATH.read_text().split("\n") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} claims to perturb")

    n_grounded = 0
    n_fallback = 0
    n_skipped = 0
    n_swap = 0
    n_swap_grounded = 0
    n_scale = 0
    n_scale_grounded = 0
    n_caption = 0
    n_caption_grounded = 0
    out_rows = []

    # Cumulative thresholds over one rng.random() draw per row -- deterministic
    # and reproducible per (pmcid, sentence), same spirit as the original
    # --swap-fraction. Order (swap, scale, caption, default cell-value) is
    # arbitrary; what matters is each slice is non-overlapping.
    #
    # "Full table altercations" (2-4 simultaneous cell edits) was implemented
    # and validated but deliberately removed here (user decision): it's the
    # heaviest of the 5 official operations in visual-token load (mean
    # post-resize area 1.25 Mpx vs 1.09-1.14 for the others) and was
    # identified as the likely cause of the v4 fine-tuning run's persistent
    # CUDA OOM on this virtualized GPU (never completed after 3 attempts). A
    # diagnostic re-render without it matched v3's (stable) memory footprint
    # almost exactly. Too risky to carry into a new fine-tuning attempt for a
    # data-diversity operation that peerj's real dev/test pairs don't even
    # use (0% observed, see module docstring) -- reintroduce only behind a
    # real fix for the OOM, not by default.
    t_swap = args.swap_fraction
    t_scale = t_swap + args.scale_fraction
    t_caption = t_scale + args.caption_fraction

    for row in rows:
        table_values = row["table_values"]
        if not table_values:
            n_skipped += 1
            continue

        rng = random.Random(hash((row["pmcid"], row["sentence"])))
        draw = rng.random()
        claim = row["claim"]

        if draw < t_swap:
            swap_result = perform_row_swap(table_values, rng, claim)
            if swap_result is not None:
                perturbed, si, sj, grounded = swap_result
                n_swap += 1
                n_swap_grounded += grounded
                out_row = dict(row)
                out_row["operation"] = "Swap rows or columns"
                out_row["perturbed_table_values"] = perturbed
                out_row["perturbation_detail"] = {
                    "row_a": si, "row_b": sj,
                    "original_row_a": table_values[si], "original_row_b": table_values[sj],
                    "grounded_in_claim": grounded,
                }
                out_rows.append(out_row)
                continue
            # No swappable row pair (table too small/irregular) -- falls
            # through to the default cell-value operation below.

        elif draw < t_scale:
            scale_result = perform_value_scaling(table_values, rng, claim)
            if scale_result is not None:
                perturbed, col, factor, grounded, changed_rows = scale_result
                n_scale += 1
                n_scale_grounded += grounded
                out_row = dict(row)
                out_row["operation"] = "Value scaling"
                out_row["perturbed_table_values"] = perturbed
                out_row["perturbation_detail"] = {
                    "col": col, "factor": factor, "changed_rows": changed_rows,
                    "grounded_in_claim": grounded,
                }
                out_rows.append(out_row)
                continue
            # No scalable numeric column -- falls through to cell-value.

        elif draw < t_caption:
            cap_result = perform_caption_adjustment(row.get("table_caption") or "", rng, claim)
            if cap_result is not None:
                perturbed_caption, grounded = cap_result
                n_caption += 1
                n_caption_grounded += grounded
                out_row = dict(row)
                out_row["operation"] = "Caption adjustments"
                out_row["perturbed_table_values"] = table_values  # image unchanged, see docstring
                out_row["perturbed_caption"] = perturbed_caption
                out_row["perturbation_detail"] = {
                    "original_caption": row.get("table_caption") or "",
                    "perturbed_caption": perturbed_caption,
                    "grounded_in_claim": grounded,
                }
                out_rows.append(out_row)
                continue
            # Caption has no specific checkable number -- falls through to
            # cell-value (this is the expected, common case, not a bug).

        # Default operation: either not selected for anything else, or the
        # selected operation's target didn't exist in this particular table
        # (fallback so the example isn't lost).
        targets = claim_target_numbers(claim)

        hit = find_claim_grounded_cell(table_values, targets) if targets else None
        grounded = hit is not None
        if hit is None:
            hit = find_random_numeric_cell(table_values, rng)
        if hit is None:
            n_skipped += 1
            continue

        i, j, span, value = hit
        raw = table_values[i][j][span[0]:span[1]]
        new_number = perturb_value(value, raw, table_values[i][j], rng)

        perturbed = [r.copy() for r in table_values]
        perturbed[i][j] = perturb_cell(table_values[i][j], span, new_number)

        if grounded:
            n_grounded += 1
        else:
            n_fallback += 1

        out_row = dict(row)
        out_row["operation"] = "Change the cell values"
        out_row["perturbed_table_values"] = perturbed
        out_row["perturbation_detail"] = {
            "row": i, "col": j,
            "original_cell": table_values[i][j],
            "perturbed_cell": perturbed[i][j],
            "grounded_in_claim": grounded,
        }
        out_rows.append(out_row)

    with args.output.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(out_rows)} tables perturbed "
          f"({n_swap} swap rows [{n_swap_grounded} grounded], "
          f"{n_scale} value scaling [{n_scale_grounded} grounded], "
          f"{n_caption} caption adjustments [{n_caption_grounded} grounded], "
          f"{n_grounded} cell-value grounded, {n_fallback} cell-value random fallback, "
          f"{n_skipped} skipped for lack of a numeric cell)")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
