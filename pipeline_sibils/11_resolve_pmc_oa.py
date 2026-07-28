#!/usr/bin/env python3
"""Step 2: resolves each PMCID to its prefix on the public PMC Open Access
S3 bucket (`s3://pmc-oa-opendata`, AWS, no authentication -- NCBI migrated
OA distribution to this bucket in early 2026; the older FTP mechanism
`oa_package/*.tar.gz` is broken/deprecated).

The version suffix (`.1`, `.2`, ...) isn't predictable ahead of time --
listed via the S3 API (`?list-type=2&prefix=<PMCID>.`) rather than guessed.
If several versions exist, the most recent one is kept.

Usage:
    python 11_resolve_pmc_oa.py --pmcid PMC2859371          # debug, single PMCID
    python 11_resolve_pmc_oa.py --input candidate_pmcids.jsonl --output pmc_oa_index.jsonl
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com/"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

CANDIDATES_PATH = Path(__file__).parent / "candidate_pmcids.jsonl"
OUTPUT_PATH = Path(__file__).parent / "pmc_oa_index.jsonl"


def list_bucket(prefix: str) -> list[dict]:
    """Objects under this prefix -- {key, size}. Single page: an article's
    file count (JATS XML, PDF, images, supplementary files) is always well
    under S3's 1000-key page limit, so pagination isn't needed here."""
    url = S3_BASE + "?" + urllib.parse.urlencode({"list-type": "2", "prefix": prefix})
    with urllib.request.urlopen(url, timeout=20) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    out = []
    for contents in root.findall(f"{S3_NS}Contents"):
        key = contents.find(f"{S3_NS}Key").text
        size = int(contents.find(f"{S3_NS}Size").text)
        out.append({"key": key, "size": size})
    return out


def resolve_pmcid(pmcid: str) -> dict | None:
    """Finds the PMC OA S3 prefix for a PMCID (highest version if several
    exist). Returns {"version", "prefix", "files": [{"key","size"}, ...]},
    or None if this PMCID has no object on the bucket at all (not in the OA
    subset -- not observed on our candidate corpus so far, but not
    guaranteed for every PMCID in general)."""
    objects = list_bucket(f"{pmcid}.")
    if not objects:
        return None

    version_re = re.compile(rf"^{re.escape(pmcid)}\.(\d+)/")
    versions = {int(m.group(1)) for o in objects if (m := version_re.match(o["key"]))}
    if not versions:
        return None
    version = max(versions)
    prefix = f"{pmcid}.{version}/"
    files = [o for o in objects if o["key"].startswith(prefix)]
    return {"version": version, "prefix": prefix, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmcid", help="Single PMCID to resolve (debug, prints and exits)")
    parser.add_argument("--input", type=Path, default=CANDIDATES_PATH,
                         help="JSONL with a 'pmcid' field per line")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.1,
                         help="Delay between requests (polite rate to the public bucket)")
    args = parser.parse_args()

    if args.pmcid:
        try:
            result = resolve_pmcid(args.pmcid)
        except urllib.error.URLError as exc:
            print(f"ERROR: {exc}")
            return
        print(json.dumps(result, indent=2) if result else f"NOT FOUND: {args.pmcid}")
        return

    rows = [json.loads(l) for l in args.input.read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]

    done: dict[str, dict] = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            r = json.loads(line)
            done[r["pmcid"]] = r

    results = list(done.values())
    n_found = sum(1 for r in results if r["found"])
    n_missing = len(results) - n_found

    for i, row in enumerate(rows):
        pmcid = row["pmcid"]
        if pmcid in done:
            continue

        try:
            resolved = resolve_pmcid(pmcid)
        except urllib.error.URLError as exc:
            print(f"[{i + 1}/{len(rows)}] {pmcid} -> ERROR {exc}, treated as not found")
            resolved = None

        if resolved:
            entry = {"pmcid": pmcid, "found": True, **resolved}
            n_found += 1
            status = f"OK v{resolved['version']} ({len(resolved['files'])} files)"
        else:
            entry = {"pmcid": pmcid, "found": False}
            n_missing += 1
            status = "NOT FOUND"
        results.append(entry)
        print(f"[{i + 1}/{len(rows)}] {pmcid} -> {status}")

        if (i + 1) % 20 == 0:
            args.output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
        if args.sleep:
            time.sleep(args.sleep)

    args.output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
    print(f"\n{n_found} found, {n_missing} absent from the PMC OA bucket, out of {len(results)} -> {args.output}")


if __name__ == "__main__":
    main()
