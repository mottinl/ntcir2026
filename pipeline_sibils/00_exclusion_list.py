#!/usr/bin/env python3
"""Step 0: list of PMCIDs to exclude from the synthetic training corpus
(anti-leakage) -- every paper_id with domain=peerj present in any official
split (dev/test, task1/task2).

IMPORTANT: paper_id is NOT a numeric PMCID. It's the native PeerJ article
number (peerj.com/articles/<paper_id>/). The DOI is 10.7717/peerj.<paper_id>,
and there is no numeric correlation between paper_id and the real PMCID
(verified: paper_id 16727 -> PMC10984177, 16850 -> PMC10984180, 19459 ->
PMC12124294). Each DOI must therefore be resolved to a PMCID via NCBI's
official converter (pmc.ncbi.nlm.nih.gov/tools/idconv), which accepts up to
200 ids per call.

The test split (task1_release.json/task2_release.json) is already public
(unlabelled, but domain/paper_id are present) at the time this script was
written -- included below alongside dev.

Usage:
    python 00_exclusion_list.py
"""

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
IDCONV_BATCH_SIZE = 200  # NCBI idconv limit

SOURCES = {
    "dev_task1": "/data/ntcir_data_train/data/dev_task1_release.json",
    "dev_task2": "/data/ntcir_data_train/data/dev_task2_release.json",
    "test_task1": "/data/ntcir_data_test/data/test_task1_release.json",
    "test_task2": "/data/ntcir_data_test/data/test_task2_release.json",
}

OUTPUT_PATH = Path(__file__).parent / "excluded_pmcids.txt"

# test_task2_release.json has no direct paper_id field (unlike
# dev_task1/dev_task2/test_task1) -- it must be derived from paper_path, e.g.
# "papers/test/peerj_19459.json" -> "19459".
PAPER_PATH_ID_RE = re.compile(r"peerj_(\d+)\.json$")


def paper_id_of(record: dict) -> str | None:
    if "paper_id" in record:
        return record["paper_id"]
    m = PAPER_PATH_ID_RE.search(record.get("paper_path", ""))
    return m.group(1) if m else None


def resolve_dois_to_pmcids(dois: list[str]) -> dict[str, str]:
    """Batch-resolve PeerJ DOIs to real PMCIDs via the NCBI idconv API.
    Returns {doi: pmcid}; DOIs the converter couldn't map are omitted.
    """
    resolved: dict[str, str] = {}
    for i in range(0, len(dois), IDCONV_BATCH_SIZE):
        batch = dois[i : i + IDCONV_BATCH_SIZE]
        qs = urllib.parse.urlencode({"ids": ",".join(batch), "format": "json"})
        with urllib.request.urlopen(f"{IDCONV_URL}?{qs}", timeout=30) as resp:
            data = json.load(resp)
        for rec in data.get("records", []):
            pmcid = rec.get("pmcid")
            if pmcid:
                resolved[rec["requested-id"]] = pmcid
            else:
                print(f"  WARNING: no PMCID for DOI {rec.get('requested-id')} "
                      f"(status={rec.get('status', 'unknown')})")
    return resolved


def main():
    paper_ids: set[str] = set()

    for name, path in SOURCES.items():
        records = json.loads(Path(path).read_text())
        source_paper_ids = set()
        n_unresolved = 0
        for r in records:
            if r.get("domain") != "peerj":
                continue
            pid = paper_id_of(r)
            if pid is None:
                n_unresolved += 1
                continue
            source_paper_ids.add(pid)
        if n_unresolved:
            print(f"  WARNING: {n_unresolved} peerj records in {name} had no resolvable paper_id")
        paper_ids |= source_paper_ids
        print(f"{name}: {len(records)} records, {len(source_paper_ids)} unique peerj paper_ids")

    print(f"\n{len(paper_ids)} unique peerj paper_ids across dev+test/task1+task2.")
    print("Resolving PeerJ DOIs (10.7717/peerj.<paper_id>) -> real PMCIDs via NCBI idconv ...")

    dois = [f"10.7717/peerj.{pid}" for pid in sorted(paper_ids)]
    doi_to_pmcid = resolve_dois_to_pmcids(dois)

    n_missing = len(dois) - len(doi_to_pmcid)
    if n_missing:
        print(f"WARNING: {n_missing}/{len(dois)} DOIs did not resolve to a PMCID "
              f"(article not in PMC, or idconv lag) -- excluded from the list below, "
              f"double check manually before trusting the exclusion filter.")

    pmcids = sorted(set(doi_to_pmcid.values()))
    print(f"\nTotal PMCIDs to exclude (resolved): {len(pmcids)}")

    OUTPUT_PATH.write_text("\n".join(pmcids) + "\n")
    print(f"-> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
