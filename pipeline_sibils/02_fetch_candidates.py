#!/usr/bin/env python3
"""Step 1: fetches the full document (body_sections, table_columns/
table_values, captions) for each PMCID in candidate_pmcids.jsonl via
`fetch?ids=<PMCID>&col=pmc`.

Caches each document under /data/pipeline_sibils_cache/raw/<pmcid>.json (not
under the repo -- expected volume ~750MB-1GB for 500 papers, based on
00_test_fetch.py's sample: ~1-1.7 MB/article). Idempotent: skips PMCIDs
already cached, so it can be resumed after an interruption.

Usage:
    python 02_fetch_candidates.py
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

FETCH_URL = "https://biodiversitypmc.sibils.org/api/fetch"
FETCH_BATCH_SIZE = 100

CANDIDATES_PATH = Path(__file__).parent / "candidate_pmcids.jsonl"
CACHE_DIR = Path("/data/pipeline_sibils_cache/raw")


def fetch_batch(pmcids: list[str]) -> dict[str, dict]:
    qs = urllib.parse.urlencode({"ids": ",".join(pmcids), "col": "pmc"})
    with urllib.request.urlopen(f"{FETCH_URL}?{qs}", timeout=120) as resp:
        data = json.load(resp)
    out = {}
    for article in data.get("sibils_article_set", []):
        doc = article.get("document", {})
        pmcid = doc.get("pmcid") or article.get("_id")
        if pmcid:
            out[pmcid] = doc
    return out


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [json.loads(l) for l in CANDIDATES_PATH.read_text().splitlines()]
    pmcids = [c["pmcid"] for c in candidates]
    print(f"{len(pmcids)} candidate PMCIDs")

    already_cached = {p.stem for p in CACHE_DIR.glob("*.json")}
    todo = [p for p in pmcids if p not in already_cached]
    print(f"{len(already_cached)} already cached, {len(todo)} to fetch")

    n_ok, n_missing = 0, 0
    for i in range(0, len(todo), FETCH_BATCH_SIZE):
        batch = todo[i : i + FETCH_BATCH_SIZE]
        try:
            docs = fetch_batch(batch)
        except urllib.error.URLError as exc:
            print(f"  WARNING: batch {i}-{i+len(batch)} failed: {exc}")
            continue

        for pmcid in batch:
            doc = docs.get(pmcid)
            if doc is None:
                n_missing += 1
                print(f"  WARNING: {pmcid} not returned by fetch")
                continue
            (CACHE_DIR / f"{pmcid}.json").write_text(json.dumps(doc, ensure_ascii=False))
            n_ok += 1

        print(f"  {min(i+FETCH_BATCH_SIZE, len(todo))}/{len(todo)} processed "
              f"({n_ok} ok, {n_missing} missing)")

    n_cached_total = len(list(CACHE_DIR.glob("*.json")))
    print(f"\nTotal cached: {n_cached_total}/{len(pmcids)} PMCIDs -> {CACHE_DIR}")


if __name__ == "__main__":
    main()
