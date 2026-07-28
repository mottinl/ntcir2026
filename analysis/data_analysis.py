#!/usr/bin/env python3
"""Basic descriptive stats on the official SciClaimEval test split (task1 +
task2): available fields, claims-per-paper, domain/evi_type/license
breakdowns, text length distributions, and a task1-vs-task2 paper overlap
check. Pure stdlib, no GPU needed -- used early on to get oriented in the
dataset before designing pipeline_baseline's prompts.

Usage:
    python data_analysis.py
"""

import json
from pathlib import Path
from collections import Counter
import statistics

DATA_DIR = Path("/data/ntcir_data_test/data")
OUTPUT_FILE = Path(__file__).resolve().parent / "data_analysis_output.txt"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log(msg="", file=None):
    if file:
        file.write(msg + "\n")


def analyze_dataset(data, name, file=None):
    log(f"\n--- {name} ---", file)

    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    log(f"Available fields: {sorted(all_keys)}", file)

    log(f"Total entries: {len(data)}", file)

    paper_ids = [item.get("paper_id", "") for item in data]
    unique_papers = set(paper_ids)
    counts = Counter(paper_ids).values()
    log(f"Unique papers: {len(unique_papers)}", file)
    log(f"Claims per paper: min={min(counts)}, max={max(counts)}, mean={len(data)/len(unique_papers):.2f}", file)

    log(f"\nDomains:", file)
    for domain, count in Counter(item.get("domain", "N/A") for item in data).most_common():
        log(f"  {domain:20s} : {count:4d} ({100*count/len(data):.1f}%)", file)

    log(f"\nEvidence types:", file)
    for evi, count in Counter(item.get("evi_type", "N/A") for item in data).most_common():
        log(f"  {evi:20s} : {count:4d} ({100*count/len(data):.1f}%)", file)

    if any("use_context" in item for item in data):
        log(f"\nuse_context:", file)
        for val, count in Counter(item.get("use_context", "N/A") for item in data).most_common():
            log(f"  {val:20s} : {count:4d} ({100*count/len(data):.1f}%)", file)

    log(f"\nText lengths (words):", file)
    for field in ["claim", "caption", "context"]:
        lengths = [len(item.get(field, "").split()) for item in data if item.get(field)]
        if lengths:
            log(f"  {field:10s} : min={min(lengths):4d}, max={max(lengths):5d}, "
                f"mean={statistics.mean(lengths):6.1f}, median={statistics.median(lengths):6.1f}", file)

    log(f"\nLicenses:", file)
    for lic, count in Counter(item.get("license_name", "N/A") for item in data).most_common():
        log(f"  {lic:30s} : {count:4d} ({100*count/len(data):.1f}%)", file)

    missing_evi = sum(1 for item in data if not item.get("evi_path"))
    claim_ids = [item.get("claim_id", "") for item in data]
    duplicates = len(claim_ids) - len(set(claim_ids))
    log(f"\nEntries without evi_path: {missing_evi}", file)
    log(f"Duplicate claim_ids     : {duplicates}", file)


def compare_datasets(task1, task2, file=None):
    log(f"\n--- Task1 vs Task2 comparison ---", file)

    papers1 = set(item.get("paper_id") for item in task1)
    papers2 = set(item.get("paper_id") for item in task2)
    log(f"Papers in common : {len(papers1 & papers2)}", file)
    log(f"Task1 only       : {len(papers1 - papers2)}", file)
    log(f"Task2 only       : {len(papers2 - papers1)}", file)

    log(f"Task1 domains    : {sorted(set(item.get('domain') for item in task1))}", file)
    log(f"Task2 domains    : {sorted(set(item.get('domain') for item in task2))}", file)


if __name__ == "__main__":
    task1 = load_json(DATA_DIR / "test_task1_release.json")
    task2 = load_json(DATA_DIR / "test_task2_release.json")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        log("NTCIR SciClaimEval data analysis", f)
        analyze_dataset(task1, "Task 1 - test_task1_release.json", f)
        analyze_dataset(task2, "Task 2 - test_task2_release.json", f)
        compare_datasets(task1, task2, f)
        log("\nAnalysis complete.", f)

    print(f"\nResults written to: {OUTPUT_FILE.resolve()}")
