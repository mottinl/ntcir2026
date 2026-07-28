#!/usr/bin/env python3
"""Step 0: first scripted call to the SIBiLS `fetch` API on a small sample
of PMCIDs, to reproducibly check the shape of the response (body_sections,
table_columns/table_values, license_ali, etc.) -- see README.md "SIBiLS API
notes" for what this call revealed.

Saves each raw response as JSON under raw/ and a readable summary to
sibils_api_notes.md.

Usage:
    python 00_test_fetch.py
"""

import json
import urllib.request
from pathlib import Path

BASE_URL = "https://biodiversitypmc.sibils.org/api/fetch"

# PMC12825236: already explored manually beforehand. The other two came
# from a real /api/search?q=table&col=pmc call (not invented IDs), to check
# that the response shape isn't an accident of the first example (presence
# of tables, body_sections structure).
TEST_PMCIDS = ["PMC12825236", "PMC9608580", "PMC4780801"]

RAW_DIR = Path(__file__).parent / "raw"
NOTES_PATH = Path(__file__).parent / "sibils_api_notes.md"


def fetch(pmcid: str) -> dict:
    url = f"{BASE_URL}?ids={pmcid}&col=pmc"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def summarize(pmcid: str, data: dict) -> str:
    lines = [f"## {pmcid}", ""]
    lines.append(f"- `success`: {data.get('success')}")
    lines.append(f"- `error`: {data.get('error')!r}")
    lines.append(f"- `warning`: {data.get('warning')!r}")

    articles = data.get("sibils_article_set") or []
    if not articles:
        lines.append("- **No article returned** (invalid PMCID, or absent from the pmc index)")
        return "\n".join(lines)

    article = articles[0]
    # The structured content lives under sibils_article_set[0]["document"],
    # not directly on the article -- confirmed empirically.
    doc = article.get("document") or {}
    lines.append(f"- `title`: {doc.get('title')!r}")
    lines.append(f"- `license_ali`: {doc.get('license_ali')!r}")
    lines.append(f"- `licence` (court): {doc.get('licence')!r}")
    lines.append(f"- `publication_date`: {doc.get('publication_date')!r}")
    lines.append(f"- `tables_in_body`: {doc.get('tables_in_body')}")
    lines.append(f"- `figures_in_body`: {doc.get('figures_in_body')}")

    body_sections = doc.get("body_sections") or []
    lines.append(f"- `body_sections`: {len(body_sections)} sections")

    n_tables = 0
    n_figures = 0
    for section in body_sections:
        for content in section.get("contents", []) or []:
            tag = content.get("tag") if isinstance(content, dict) else None
            if tag == "table":
                n_tables += 1
                cols = content.get("table_columns")
                vals = content.get("table_values")
                lines.append(
                    f"  - table `{content.get('xref_id')}` ({content.get('label')}): "
                    f"{len(cols) if cols else 0} columns, "
                    f"{len(vals) if vals else 0} value rows"
                )
            elif tag == "fig":
                n_figures += 1
                lines.append(
                    f"  - figure `{content.get('xref_id')}` ({content.get('label')}): "
                    f"caption={content.get('caption', '')[:80]!r}, "
                    f"graphics={content.get('graphics')}"
                )
    lines.append(f"- Total (counted from body_sections): {n_tables} tables, {n_figures} figures")
    return "\n".join(lines)


def main():
    RAW_DIR.mkdir(exist_ok=True)
    summaries = ["# SIBiLS API notes -- output of 00_test_fetch.py", ""]

    for pmcid in TEST_PMCIDS:
        print(f"Fetching {pmcid} ...")
        try:
            data = fetch(pmcid)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            summaries.append(f"## {pmcid}\n\n- **Request failed**: {exc!r}")
            continue

        raw_path = RAW_DIR / f"{pmcid}.json"
        raw_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  saved raw -> {raw_path}")

        summaries.append(summarize(pmcid, data))
        summaries.append("")

    NOTES_PATH.write_text("\n".join(summaries))
    print(f"\nNotes -> {NOTES_PATH}")


if __name__ == "__main__":
    main()
