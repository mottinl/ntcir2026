#!/usr/bin/env python3
"""Step 0: first list of candidate biomedical PMCIDs for the step-1 pilot.

Strategy (REST search+fetch, good enough for a pilot -- direct Elasticsearch
access would be the option to prefer if this ever needs to scale past a few
thousand papers):

1. Three query "buckets" (for topical diversity in the future training set,
   while staying 100% biomedical/PMC -- no extension to arXiv/ACL):
   - "biomed"    : no keyword constraint (general population of `col=pmc`,
                   already a biomedical corpus by construction)
   - "biomed_ml" : biomedical papers touching machine learning
   - "biomed_nlp": biomedical papers touching NLP / text mining
2. License filter on the ES side (`terms` on the `licence` field): an
   explicit allowlist excluding ND variants and absent/unclear licenses
   (`None`, `"NO-CC CODE"`).
3. Exclusion of the PMCIDs in `excluded_pmcids.txt` (dev/test anti-leakage).
4. Actual table-presence check via `fetch` (`document.tables_in_body` field,
   the only reliable signal -- the `search` index's `tables` field exists
   even when empty, tested and rejected) -- keeps only `tables_in_body > 0`.

Usage:
    python 01_query_sibils.py
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SEARCH_URL = "https://biodiversitypmc.sibils.org/api/search"
FETCH_URL = "https://biodiversitypmc.sibils.org/api/fetch"

# Excludes ND variants (No Derivatives, incompatible with programmatically
# perturbing the evidence) and absent/unclear licenses. NC/SA are kept: the
# task organizers themselves use CC BY-NC and CC BY-NC-SA in the official
# dataset.
ALLOWED_LICENCES = ["CC BY", "CC BY-NC", "CC BY-NC-SA", "CC BY-SA", "CC0"]

# Per bucket: how many PMCIDs to request from `search` (before the table
# filter) and the target count to keep after verifying tables_in_body>0 via
# `fetch`. v2: targets doubled (200/150/150 -> 400/300/300, same 2:1
# search:target ratio) to grow the corpus while keeping the same topical split.
BUCKETS = {
    "biomed": {
        "query": None,  # no text constraint -- general population of col=pmc
        "n_search": 800,
        "target": 400,
    },
    "biomed_ml": {
        "query": "machine learning",
        "n_search": 600,
        "target": 300,
    },
    "biomed_nlp": {
        "query": "natural language processing",
        "n_search": 600,
        "target": 300,
    },
}

FETCH_BATCH_SIZE = 200  # well under the documented 1000/call cap, keeps individual requests fast

OUTPUT_PATH = Path(__file__).parent / "candidate_pmcids.jsonl"
EXCLUDED_PATH = Path(__file__).parent / "excluded_pmcids.txt"


def build_jq(query_phrase: str | None) -> str:
    filters = [{"terms": {"licence": ALLOWED_LICENCES}}]
    if query_phrase:
        must = [{"multi_match": {"query": query_phrase, "type": "phrase",
                                  "fields": ["title", "abstract", "full_text"]}}]
        body = {"query": {"bool": {"must": must, "filter": filters}}}
    else:
        body = {"query": {"bool": {"filter": filters}}}
    return json.dumps(body)


def search_pmcids(query_phrase: str | None, n: int) -> list[str]:
    qs = urllib.parse.urlencode({"jq": build_jq(query_phrase), "col": "pmc", "n": n})
    with urllib.request.urlopen(f"{SEARCH_URL}?{qs}", timeout=60) as resp:
        data = json.load(resp)
    hits = data.get("elastic_output", {}).get("hits", {}).get("hits", [])
    return [h["_id"] for h in hits]


def fetch_documents(pmcids: list[str]) -> dict[str, dict]:
    """Returns {pmcid: document} for whichever ids resolved."""
    out: dict[str, dict] = {}
    for i in range(0, len(pmcids), FETCH_BATCH_SIZE):
        batch = pmcids[i : i + FETCH_BATCH_SIZE]
        qs = urllib.parse.urlencode({"ids": ",".join(batch), "col": "pmc"})
        try:
            with urllib.request.urlopen(f"{FETCH_URL}?{qs}", timeout=90) as resp:
                data = json.load(resp)
        except urllib.error.URLError as exc:
            print(f"  WARNING: fetch batch {i}-{i+len(batch)} failed: {exc}")
            continue
        for article in data.get("sibils_article_set", []):
            doc = article.get("document", {})
            pmcid = doc.get("pmcid") or article.get("_id")
            if pmcid:
                out[pmcid] = doc
    return out


def main():
    excluded = set(EXCLUDED_PATH.read_text().split())
    print(f"{len(excluded)} PMCIDs exclus (dev/test)")

    seen: set[str] = set()
    candidates: list[dict] = []

    for bucket_name, cfg in BUCKETS.items():
        print(f"\n=== bucket {bucket_name!r} (query={cfg['query']!r}) ===")
        pmcids = search_pmcids(cfg["query"], cfg["n_search"])
        pmcids = [p for p in pmcids if p not in excluded and p not in seen]
        print(f"  {len(pmcids)} candidate PMCIDs (search, deduplicated, excluded ones removed)")

        docs = fetch_documents(pmcids)
        print(f"  {len(docs)}/{len(pmcids)} resolved via fetch")

        kept = 0
        for pmcid in pmcids:
            doc = docs.get(pmcid)
            if not doc or doc.get("tables_in_body", 0) <= 0:
                continue
            seen.add(pmcid)
            candidates.append({
                "pmcid": pmcid,
                "bucket": bucket_name,
                "title": doc.get("title"),
                "licence": doc.get("licence"),
                "doi": doc.get("doi"),
                "tables_in_body": doc.get("tables_in_body"),
                "figures_in_body": doc.get("figures_in_body"),
                "publication_date": doc.get("publication_date"),
            })
            kept += 1
            if kept >= cfg["target"]:
                break
        print(f"  {kept} kept (tables_in_body > 0, target {cfg['target']})")

    print(f"\nTotal candidates kept (all buckets, deduplicated): {len(candidates)}")

    with OUTPUT_PATH.open("w") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
