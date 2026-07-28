#!/usr/bin/env python3
"""Step 2: downloads real figure images from PMC OA (S3 bucket
`pmc-oa-opendata`, see `11_resolve_pmc_oa.py`) for every figure listed in
the already-cached SIBiLS documents
(`/data/pipeline_sibils_cache/raw/<pmcid>.json`).

For each figure (`contents[i]` with `tag=="fig"`), SIBiLS' `graphics` field
gives the base filename (no extension) -- confirmed identical to the real
filename on PMC OA (verified on PMC2859371 and cross-checked via Europe
PMC's XML). This name is looked up in `pmc_oa_index.jsonl` (already
resolved per PMCID) to recover the real extension and download the binary.

Usage:
    python 12_fetch_figures.py --limit 5   # smoke test
    python 12_fetch_figures.py             # full corpus (500 candidates)
"""

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com/"

RAW_DIR = Path("/data/pipeline_sibils_cache/raw")
FIGURES_DIR = Path("/data/pipeline_sibils_cache/figures")
CANDIDATES_PATH = Path(__file__).parent / "candidate_pmcids.jsonl"
INDEX_PATH = Path(__file__).parent / "pmc_oa_index.jsonl"
OUTPUT_PATH = Path(__file__).parent / "figure_mentions.jsonl"
STATE_PATH = Path(__file__).parent / "fetch_figures_done.txt"

# Preference order if a graphics basename somehow matches more than one file
# (not observed in practice -- PMC OA only ships the main image, no
# Europe-PMC-style .gif thumbnail duplicate -- but cheap to guard against).
IMAGE_EXT_PRIORITY = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif"]


def load_pmc_oa_index() -> dict[str, dict]:
    index = {}
    for line in INDEX_PATH.read_text().splitlines():
        r = json.loads(line)
        if r["found"]:
            index[r["pmcid"]] = r
    return index


def find_figure_file(oa_entry: dict, graphics_basename: str) -> dict | None:
    prefix = oa_entry["prefix"]
    candidates = []
    for f in oa_entry["files"]:
        key = f["key"]
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        stem = name.rsplit(".", 1)[0] if "." in name else name
        if stem == graphics_basename:
            candidates.append(f)
    if not candidates:
        return None

    def ext_rank(f):
        key = f["key"]
        ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
        return IMAGE_EXT_PRIORITY.index(ext) if ext in IMAGE_EXT_PRIORITY else len(IMAGE_EXT_PRIORITY)

    candidates.sort(key=ext_rank)
    return candidates[0]


def download(key: str, dest: Path):
    url = S3_BASE + urllib.parse.quote(key)
    with urllib.request.urlopen(url, timeout=30) as resp:
        dest.write_bytes(resp.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pmc_oa_index = load_pmc_oa_index()

    rows = [json.loads(l) for l in args.candidates.read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]

    done_pmcids: set[str] = set()
    if STATE_PATH.exists():
        done_pmcids = set(STATE_PATH.read_text().splitlines())

    out_rows: list[dict] = []
    if args.output.exists():
        out_rows = [json.loads(l) for l in args.output.read_text().splitlines()]

    n_figures = n_downloaded = n_no_oa = n_no_match = n_dl_error = 0

    for i, row in enumerate(rows):
        pmcid = row["pmcid"]
        if pmcid in done_pmcids:
            continue

        raw_path = RAW_DIR / f"{pmcid}.json"
        if not raw_path.exists():
            print(f"[{i + 1}/{len(rows)}] {pmcid}: no cached SIBiLS document, skipped")
            continue
        # 02_fetch_candidates.py caches the document sub-object directly (not
        # the full fetch response wrapper), so no "document" key to unwrap here.
        doc = json.loads(raw_path.read_text())

        oa_entry = pmc_oa_index.get(pmcid)
        if oa_entry is None:
            n_no_oa += 1
            done_pmcids.add(pmcid)
            print(f"[{i + 1}/{len(rows)}] {pmcid}: absent from pmc_oa_index, skipped")
            continue

        n_this_article = 0
        for section in doc.get("body_sections", []):
            for c in section.get("contents", []):
                if c.get("tag") != "fig":
                    continue
                graphics = c.get("graphics") or []
                if not graphics:
                    continue
                basename = graphics[0]
                n_figures += 1

                match = find_figure_file(oa_entry, basename)
                if match is None:
                    n_no_match += 1
                    continue

                filename = match["key"].rsplit("/", 1)[-1]
                dest = args.out_dir / f"{pmcid}_{filename}"
                if not dest.exists():
                    try:
                        download(match["key"], dest)
                    except (urllib.error.URLError, OSError) as exc:
                        print(f"  [{pmcid}] download failed for {match['key']}: {exc}")
                        n_dl_error += 1
                        continue

                n_downloaded += 1
                n_this_article += 1
                out_rows.append({
                    "pmcid": pmcid,
                    "label": c.get("label"),
                    "caption": c.get("caption"),
                    "xref_id": c.get("xref_id"),
                    "graphics": basename,
                    "s3_key": match["key"],
                    "local_path": str(dest),
                })

        done_pmcids.add(pmcid)
        print(f"[{i + 1}/{len(rows)}] {pmcid}: {n_this_article} figures downloaded")

        if (i + 1) % 20 == 0:
            args.output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n")
            STATE_PATH.write_text("\n".join(sorted(done_pmcids)) + "\n")

    args.output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n")
    STATE_PATH.write_text("\n".join(sorted(done_pmcids)) + "\n")
    print(f"\n{n_downloaded}/{n_figures} figures downloaded "
          f"({n_no_oa} PMCIDs absent from pmc_oa_index, {n_no_match} with no matching file, "
          f"{n_dl_error} network errors) -> {args.output}")


if __name__ == "__main__":
    main()
