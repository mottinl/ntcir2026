#!/usr/bin/env python3
"""Step 2: perturbs the evidence (figure image) to generate the Refuted
counterpart of each figure claim -- same claim, modified figure (same
principle as 05_perturb_table.py for tables).

Official figure-edit taxonomy: "Legend Swap, Graph Flip, Graph Swap, and
others". Verified by diffing real Supported/Refuted pairs from the dev set
(peerj domain, 97 figure examples) -- real distribution: Category Swap 36
(37%), Legend Swap 30 (31%), Graph Swap 8 (8%), Graph Flip 6 (6%),
Supported_claim_only 17. Contrary to the initial assumption, "Graph Flip" is
NOT a mirror-flip of the image -- diffed pixel by pixel, it entirely
replaces a data series' values (requires the chart's underlying data, out
of scope for the "direct image manipulation" approach chosen here). The
other 3 operations (76% of the real distribution) are, by contrast, pure
image manipulations (cropping/rearranging/swapping pixel regions already
present, no generation):

- **"Graph Swap"** (implemented here): reorders sub-panels within a
  multi-panel figure (e.g. A/B/C/D), each panel's content unchanged.
  Detected via whitespace gutters (rows/columns nearly free of ink): first
  a split into horizontal bands; if ALL bands re-split into columns
  consistently (same number of columns everywhere), it's a real NxM grid
  (e.g. an 8-scatter-plot figure laid out 4x2) and each cell is a panel;
  otherwise (inconsistent split widths across bands -- typically A/B/C
  letter panels each with their own title/legend/axis) each whole band is
  treated as a single panel, so as not to fragment one logical panel into
  pieces (a bug found while testing on real figures: the naive split was
  isolating axis labels and titles as false separate "panels"). Only
  near-identical-size panels (`--size-tolerance`) are swapped, to avoid any
  visible distortion.
  **Measured coverage on the real corpus (3665 figures): 951 (26%)** have
  at least 2 swappable panels -- the rest (single-panel figure, or panels
  too different in size) simply aren't eligible for this operation (not an
  error, just out of scope, like "Caption adjustments" for tables).

- **"Category Swap"** (implemented here, fallback when "Graph Swap" doesn't
  apply): swaps two EXISTING text labels in the figure (e.g. two gene names
  in the same label column) -- OCR detection (`easyocr`) then groups text
  boxes into "families" (same font size, row- or column-aligned), swapping
  two different, comparably-sized text boxes within the same family.
  Purely numeric boxes (axis values, p-values) are excluded from grouping
  -- found while testing on the real official example (Category Swap,
  PMC/paper_id 16935): without this filter, a scale value ("4962") was
  pairing with a real gene name ("ETS2"), producing a nonsensical pair.
  Validated: on that same official example, the ETS2/FOS/IL1A label family
  (a gene-name column of an oncoprint) is correctly detected.
  **CUDA_VISIBLE_DEVICES forced to "1"** (GPU 1, free -- distinct from GPU
  0 used by the claim normalization in progress and GPU 2 used by another
  task): CPU OCR measured ~80s/image, i.e. ~32h for the whole corpus -- far
  too costly for what it's worth (decision: abandon rather than pay that
  cost). On GPU, measured at ~0.2-1.4s/image, so ~15-20min for the whole
  corpus -- changes the decision: implemented with this constraint.

- **"Legend Swap"**: not yet implemented.

Usage:
    python 14_perturb_figures.py [--limit N]
"""

import argparse
import json
import os
import random
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from PIL import Image

INPUT_PATH = Path(__file__).parent / "figure_claims.jsonl"
OUTPUT_PATH = Path(__file__).parent / "perturbed_figure_claims.jsonl"
PERTURBED_DIR = Path("/data/pipeline_sibils_cache/figures_perturbed")

OCR_MIN_CONF = 0.35
OCR_HEIGHT_TOL = 0.20   # relative tolerance on box height (font size proxy)
OCR_WIDTH_TOL = 0.30    # relative tolerance on box width (needed for a clean swap)
OCR_ALIGN_TOL_PX = 12   # tolerance for "same row" (close center_y) / "same
                        # column" (close left-x)

INK_THRESHOLD = 245  # grayscale value below which a pixel counts as "ink"
MIN_GAP_FRAC = 0.012  # minimum gutter width, as a fraction of that axis' length
MIN_GAP_PX = 6
MIN_PANEL_FRAC = 0.10  # a band smaller than this fraction of the axis is
                       # furniture (axis label sliver, legend strip), not a
                       # real panel
DEFAULT_SIZE_TOLERANCE = 0.15  # max relative (w, h) difference to consider
                                # two panels swappable without visible distortion


def _find_gutters(ink_mask_1d: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Returns (start, end) index ranges of pure-whitespace runs >= min_gap,
    restricted to the [first_ink, last_ink] content bounding box on this axis
    (so leading/trailing page whitespace isn't itself treated as a gutter)."""
    ink_idx = np.where(ink_mask_1d)[0]
    if len(ink_idx) == 0:
        return []
    lo, hi = ink_idx[0], ink_idx[-1]
    gutters = []
    run_start = None
    for i in range(lo, hi + 1):
        if not ink_mask_1d[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_gap:
                gutters.append((run_start, i))
            run_start = None
    if run_start is not None and (hi + 1 - run_start) >= min_gap:
        gutters.append((run_start, hi + 1))
    return gutters


def _segments_from_gutters(length: int, gutters: list[tuple[int, int]]) -> list[tuple[int, int]]:
    segs = []
    prev = 0
    for (gs, ge) in gutters:
        if gs > prev:
            segs.append((prev, gs))
        prev = ge
    if prev < length:
        segs.append((prev, length))
    return segs


def _filter_small_bands(bands: list[tuple[int, int]], length: int,
                         min_frac: float = MIN_PANEL_FRAC) -> list[tuple[int, int]]:
    min_len = length * min_frac
    return [(a, b) for (a, b) in bands if (b - a) >= min_len]


def detect_panels(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """Returns a list of (x0, y0, x1, y1) panel bounding boxes."""
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    ink = gray < INK_THRESHOLD

    row_has_ink = ink.any(axis=1)
    min_gap_row = max(MIN_GAP_PX, int(h * MIN_GAP_FRAC))
    row_gutters = _find_gutters(row_has_ink, min_gap_row)
    row_bands = _filter_small_bands(_segments_from_gutters(h, row_gutters), h)
    if not row_bands:
        return []

    min_gap_col = max(MIN_GAP_PX, int(w * MIN_GAP_FRAC))
    per_row_cols = []
    for (ry0, ry1) in row_bands:
        band_ink = ink[ry0:ry1, :]
        col_has_ink = band_ink.any(axis=0)
        col_gutters = _find_gutters(col_has_ink, min_gap_col)
        col_segs = _filter_small_bands(_segments_from_gutters(w, col_gutters), w)
        per_row_cols.append(col_segs)

    n_cols = len(per_row_cols[0])
    uniform_grid = n_cols >= 2 and all(len(cs) == n_cols for cs in per_row_cols)

    panels = []
    if uniform_grid:
        for (ry0, ry1), col_segs in zip(row_bands, per_row_cols):
            for (cx0, cx1) in col_segs:
                panels.append((cx0, ry0, cx1, ry1))
    else:
        for (ry0, ry1) in row_bands:
            panels.append((0, ry0, w, ry1))
    return panels


def swappable_pairs(panels: list[tuple[int, int, int, int]],
                     size_tol: float = DEFAULT_SIZE_TOLERANCE) -> list[tuple[int, int]]:
    """Pairs of panel indices with near-identical (w, h) -- only these are
    safe to swap without visible resizing distortion."""
    dims = [(x1 - x0, y1 - y0) for (x0, y0, x1, y1) in panels]
    pairs = []
    for i in range(len(panels)):
        for j in range(i + 1, len(panels)):
            w1, h1 = dims[i]
            w2, h2 = dims[j]
            if abs(w1 - w2) <= size_tol * max(w1, w2) and abs(h1 - h2) <= size_tol * max(h1, h2):
                pairs.append((i, j))
    return pairs


def perform_graph_swap(img: Image.Image, rng: random.Random):
    """Returns (perturbed_image, (panel_a, panel_b)) or None if this image
    has no swappable panel pair."""
    panels = detect_panels(img)
    pairs = swappable_pairs(panels)
    if not pairs:
        return None
    i, j = rng.choice(pairs)
    x0a, y0a, x1a, y1a = panels[i]
    x0b, y0b, x1b, y1b = panels[j]
    crop_a = img.crop((x0a, y0a, x1a, y1a))
    crop_b = img.crop((x0b, y0b, x1b, y1b))
    out = img.copy()
    out.paste(crop_b.resize((x1a - x0a, y1a - y0a)), (x0a, y0a))
    out.paste(crop_a.resize((x1b - x0b, y1b - y0b)), (x0b, y0b))
    return out, (panels[i], panels[j])


def _ocr_boxes(reader, path):
    result = reader.readtext(str(path))
    boxes = []
    for bbox, text, conf in result:
        if conf < OCR_MIN_CONF or not text.strip():
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        boxes.append({
            "text": text, "conf": conf,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "w": x1 - x0, "h": y1 - y0,
            "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
        })
    return boxes


def _looks_numeric(text: str) -> bool:
    """True for axis tick values / stats (e.g. "4962", "0.044", "12%") --
    these aren't category labels, and swapping one against a real label (e.g.
    a gene name) produces a nonsense pair, not a valid Category Swap."""
    stripped = text.strip().strip("%").replace(",", "").replace(".", "").replace("-", "")
    return stripped.isdigit()


def find_category_swap_candidates(boxes):
    """Groups OCR boxes into 'families' (row-aligned or column-aligned,
    similar font size and width) and returns (i, j) pairs of DIFFERENT,
    non-numeric text boxes within a family -- safe to swap as existing-label
    patches without visible distortion."""
    pairs = []
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if a["text"].strip().lower() == b["text"].strip().lower():
                continue  # identical text -- a no-op swap, not useful
            if _looks_numeric(a["text"]) or _looks_numeric(b["text"]):
                continue
            if abs(a["h"] - b["h"]) > OCR_HEIGHT_TOL * max(a["h"], b["h"]):
                continue
            if abs(a["w"] - b["w"]) > OCR_WIDTH_TOL * max(a["w"], b["w"]):
                continue
            same_row = abs(a["cy"] - b["cy"]) <= OCR_ALIGN_TOL_PX
            same_col = abs(a["x0"] - b["x0"]) <= OCR_ALIGN_TOL_PX
            if same_row or same_col:
                pairs.append((i, j))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--size-tolerance", type=float, default=DEFAULT_SIZE_TOLERANCE)
    parser.add_argument("--no-ocr", action="store_true",
                         help="Skip Category Swap entirely (Graph Swap only).")
    args = parser.parse_args()

    rows = [json.loads(l) for l in INPUT_PATH.read_text().split("\n") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} figure claims to perturb")

    PERTURBED_DIR.mkdir(parents=True, exist_ok=True)

    ocr_reader = None
    if not args.no_ocr:
        import easyocr
        print("Loading easyocr (GPU 1) ...")
        ocr_reader = easyocr.Reader(["en"], gpu=True)
        print("easyocr loaded.")

    n_graph_swap = 0
    n_category_swap = 0
    n_not_applicable = 0
    n_missing_image = 0
    out_rows = []

    for idx, row in enumerate(rows):
        local_path = Path(row["local_path"])
        if not local_path.exists():
            n_missing_image += 1
            continue

        rng = random.Random(hash((row["pmcid"], row["sentence"])))
        img = Image.open(local_path).convert("RGB")

        operation = None
        perturbed_img = None
        detail = None

        result = perform_graph_swap(img, rng)
        if result is not None:
            perturbed_img, (panel_a, panel_b) = result
            operation = "Graph Swap"
            detail = {"panel_a": panel_a, "panel_b": panel_b}
        elif ocr_reader is not None:
            boxes = _ocr_boxes(ocr_reader, local_path)
            pairs = find_category_swap_candidates(boxes)
            if pairs:
                i, j = rng.choice(pairs)
                a, b = boxes[i], boxes[j]
                box_a = (int(a["x0"]), int(a["y0"]), int(a["x1"]), int(a["y1"]))
                box_b = (int(b["x0"]), int(b["y0"]), int(b["x1"]), int(b["y1"]))
                crop_a = img.crop(box_a)
                crop_b = img.crop(box_b)
                perturbed_img = img.copy()
                perturbed_img.paste(crop_b.resize((box_a[2] - box_a[0], box_a[3] - box_a[1])), (box_a[0], box_a[1]))
                perturbed_img.paste(crop_a.resize((box_b[2] - box_b[0], box_b[3] - box_b[1])), (box_b[0], box_b[1]))
                operation = "Category Swap"
                detail = {"box_a": box_a, "box_b": box_b, "text_a": a["text"], "text_b": b["text"]}

        if operation is None:
            n_not_applicable += 1
            continue

        if operation == "Graph Swap":
            n_graph_swap += 1
        else:
            n_category_swap += 1

        out_name = f"{row['pmcid']}_{idx}_refuted{local_path.suffix}"
        out_path = PERTURBED_DIR / out_name
        perturbed_img.save(out_path)

        out_row = dict(row)
        out_row["operation"] = operation
        out_row["perturbed_local_path"] = str(out_path)
        out_row["perturbation_detail"] = detail
        out_rows.append(out_row)

    with args.output.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(out_rows)} figures perturbed "
          f"({n_graph_swap} Graph Swap, {n_category_swap} Category Swap)")
    print(f"{n_not_applicable} with no applicable operation (not an error, out of scope)")
    if n_missing_image:
        print(f"{n_missing_image} local images missing (skipped)")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
