#!/usr/bin/env python3
"""Step 1: renders `table_values`/`table_columns` (original and perturbed)
to PNG, in a style visually close to the official dataset's
`tables_png/dev/*.png` (light-blue caption band, blue-underlined header,
rows separated by a gray rule -- see `tables/dev/val_tab_0417.html` for the
official CSS template).

No HTML rendering engine is available in this environment (no weasyprint/
playwright/wkhtmltoimage, only PIL) -- drawn directly with `ImageDraw`.
DejaVu Serif (system font), in the same spirit as the official template's
Georgia.

v2: width changed from 1024 to 1655px to match the official dev set PNGs'
real width (a ~1.6x gap was measured, which could shift the model's visual
tokenization between training and evaluation). Also fixed a rendering bug
found during the same diagnostic: on tables with many columns (13.6% of the
v2 corpus has more than 12 columns), the old column-width algorithm shrank
columns proportionally then applied a 50px floor WITHOUT re-checking that
the total still fit in the image -- combined with `_wrap()` letting an
overly long single word overflow its column (the only way not to cut it),
one column's text visually "bled" into the neighboring column (illegible
headers, confirmed by a zero-shot OCR transcription test that made the
model hallucinate in a loop on these images). Two fixes: (1) the image
width now grows dynamically for many-column tables instead of squeezing
columns below a readable threshold (capped at `MAX_WIDTH` so the visual
token budget doesn't explode on extreme cases, up to 119 columns observed);
(2) `_wrap()` falls back to a character-by-character break if a single word
still exceeds a column's width, instead of letting it overflow as-is.

For each row of `perturbed_claims.jsonl`, generates two images:
`<pmcid>_<idx>_supported.png` (original evidence) and
`<pmcid>_<idx>_refuted.png` (perturbed evidence) -- same claim for both,
mirroring the official dataset's own Supported/Refuted construction.

Usage:
    python 06_render_table_png.py --limit 5 --out-dir tables_png_preview
    python 06_render_table_png.py --out-dir /data/pipeline_sibils_cache/tables_png
"""

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

INPUT_PATH = Path(__file__).parent / "perturbed_claims.jsonl"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REGULAR = FONT_DIR / "DejaVuSerif.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSerif-Bold.ttf"
FONT_ITALIC = FONT_DIR / "DejaVuSerif-Italic.ttf"

WIDTH = 1655  # matches the official dev set tables_png/*.png width (was 1024, see module docstring)
MAX_WIDTH = 2400  # hard cap for many-column tables so vision-token count stays bounded
PAD = 20
ROW_PAD_Y = 8
CELL_PAD_X = 12
MIN_COL_W = 90
MAX_COL_W = 320
FONT_SIZE = 15
CAPTION_FONT_SIZE = 16

CAPTION_BG = (234, 243, 251)
BLUE = (0, 122, 204)
GRAY_BORDER = (204, 204, 204)
BLACK = (20, 20, 20)


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _clean(text) -> str:
    """Collapse embedded newlines/runs of whitespace to single spaces --
    PIL's textlength() raises on multiline text, and some SIBiLS-parsed
    cells contain literal "\\n" (e.g. a dataset-split breakdown cell),
    found via a crash when rendering the full corpus."""
    return " ".join(str(text).split())


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if not text:
        return [""]
    max_width = max(max_width, 1)
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
        # A single word (or the first token on a fresh line) can still be
        # wider than max_width on a narrow column -- break it at the
        # character level instead of letting it overflow uncut, which used
        # to bleed visually into the next column on many-column tables
        # (this was the actual mechanism behind the
        # illegible overlapping headers, not just a cosmetic wrap issue).
        while draw.textlength(cur, font=font) > max_width and len(cur) > 1:
            split = len(cur)
            while split > 1 and draw.textlength(cur[:split], font=font) > max_width:
                split -= 1
            lines.append(cur[:split])
            cur = cur[split:]
    if cur:
        lines.append(cur)
    return lines or [""]


def _col_widths(draw, columns: list[str], values: list[list[str]],
                 font_reg, font_bold, n_cols: int) -> tuple[list[int], int]:
    """Returns (column widths, image width). Grows the image width for
    many-column tables instead of squeezing columns below a readable
    minimum (the old fixed-WIDTH scale-down had an unclamped 50px floor
    that could still overflow the canvas on wide tables, and even where it
    fit, 50px is often too narrow for a single unbreakable header word --
    see module docstring)."""
    widths = []
    for c in range(n_cols):
        header = _clean(columns[c]) if c < len(columns) else ""
        w = draw.textlength(header, font=font_bold)
        for row in values:
            cell = _clean(row[c]) if c < len(row) else ""
            w = max(w, draw.textlength(cell, font=font_reg))
        widths.append(int(min(max(w + 2 * CELL_PAD_X, MIN_COL_W), MAX_COL_W)))
    total = sum(widths)
    avail = WIDTH - 2 * PAD

    if total <= avail:
        # Spread the leftover space proportionally so the table still
        # spans the full canvas width (matches the official renderer's
        # look, which doesn't leave a ragged blank strip on the right).
        if total > 0:
            extra = avail - total
            widths = [w + int(extra * w / total) for w in widths]
        return widths, WIDTH

    needed = total + 2 * PAD
    if needed <= MAX_WIDTH:
        return widths, needed

    # Extreme case (up to 119 columns observed in the v2 corpus): still
    # shrink toward MIN_COL_W within MAX_WIDTH, but _wrap()'s
    # character-level fallback keeps any single long word from bleeding
    # into the next column even at this reduced width.
    avail = MAX_WIDTH - 2 * PAD
    scale = avail / total
    widths = [max(int(w * scale), MIN_COL_W // 2) for w in widths]
    return widths, MAX_WIDTH


def render_table(table_label: str, caption: str, columns: list[str],
                  values: list[list[str]]) -> Image.Image:
    table_label, caption = _clean(table_label), _clean(caption)
    font_reg = _font(FONT_REGULAR, FONT_SIZE)
    font_bold = _font(FONT_BOLD, FONT_SIZE)
    font_cap = _font(FONT_BOLD, CAPTION_FONT_SIZE)
    font_label = _font(FONT_BOLD, CAPTION_FONT_SIZE)

    n_cols = max(len(columns), max((len(r) for r in values), default=0))
    scratch = Image.new("RGB", (WIDTH, 100), "white")
    draw = ImageDraw.Draw(scratch)
    col_w, img_width = _col_widths(draw, columns, values, font_reg, font_bold, n_cols)

    # -- pass 1: measure total height --
    cap_lines = _wrap(draw, f"{table_label}  {caption}".strip(), font_cap, img_width - 2 * PAD)
    y = PAD + len(cap_lines) * (CAPTION_FONT_SIZE + 6) + 16

    def row_height(cells: list[str], font) -> int:
        n_lines = 1
        for c in range(n_cols):
            text = _clean(cells[c]) if c < len(cells) else ""
            n_lines = max(n_lines, len(_wrap(draw, text, font, col_w[c] - 2 * CELL_PAD_X)))
        return n_lines * (FONT_SIZE + 6) + 2 * ROW_PAD_Y

    header_h = row_height(columns, font_bold)
    y += header_h
    row_heights = [row_height(row, font_reg) for row in values]
    y += sum(row_heights) + PAD

    img = Image.new("RGB", (img_width, int(y)), "white")
    draw = ImageDraw.Draw(img)

    # -- caption bar --
    cap_h = len(cap_lines) * (CAPTION_FONT_SIZE + 6) + 16
    draw.rectangle([0, 0, img_width, cap_h], fill=CAPTION_BG)
    ty = 8
    if cap_lines:
        draw.text((PAD, ty), table_label, font=font_label, fill=BLUE)
        label_w = draw.textlength(table_label + "  ", font=font_label)
        draw.text((PAD + label_w, ty), cap_lines[0][len(table_label):].lstrip() if cap_lines[0].startswith(table_label) else cap_lines[0],
                   font=font_cap, fill=BLACK)
        ty += CAPTION_FONT_SIZE + 6
        for line in cap_lines[1:]:
            draw.text((PAD, ty), line, font=font_cap, fill=BLACK)
            ty += CAPTION_FONT_SIZE + 6

    cur_y = cap_h + 12

    def draw_row(cells: list[str], font, fill, h: int, top_border=None, bottom_border=None):
        nonlocal cur_y
        if top_border:
            draw.line([(PAD, cur_y), (img_width - PAD, cur_y)], fill=top_border[0], width=top_border[1])
        x = PAD
        for c in range(n_cols):
            text = _clean(cells[c]) if c < len(cells) else ""
            lines = _wrap(draw, text, font, col_w[c] - 2 * CELL_PAD_X)
            ly = cur_y + ROW_PAD_Y
            for line in lines:
                draw.text((x + CELL_PAD_X, ly), line, font=font, fill=fill)
                ly += FONT_SIZE + 6
            x += col_w[c]
        cur_y += h
        if bottom_border:
            draw.line([(PAD, cur_y), (img_width - PAD, cur_y)], fill=bottom_border[0], width=bottom_border[1])

    draw_row(columns, font_bold, BLACK, header_h, bottom_border=(BLUE, 3))
    for row, h in zip(values, row_heights):
        draw_row(row, font_reg, BLACK, h, bottom_border=(GRAY_BORDER, 1))

    return img.crop((0, 0, img_width, cur_y + PAD))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # str.splitlines() wrongly breaks on real U+2028/U+2029 characters
    # found inside a table cell (e.g. PMC2874376: "MLST<U+2028>ST") -- strict
    # split here (same fix as in 04_normalize_claims.py).
    rows = [json.loads(l) for l in args.input.read_text().split("\n") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    for idx, row in enumerate(rows):
        label = row.get("table_label") or ""
        caption = row.get("table_caption") or ""
        columns = row.get("table_columns") or []

        img_sup = render_table(label, caption, columns, row["table_values"])
        img_sup.save(out_dir / f"{row['pmcid']}_{idx}_supported.png")

        img_ref = render_table(label, caption, columns, row["perturbed_table_values"])
        img_ref.save(out_dir / f"{row['pmcid']}_{idx}_refuted.png")

        print(f"[{idx+1}/{len(rows)}] {row['pmcid']}")

    print(f"-> {out_dir} ({2 * len(rows)} images)")


if __name__ == "__main__":
    main()
