"""
pipeline.py
-----------
End-to-end pipeline: PubMed search → LLM extraction → structured output.

Runs fetch_pubmed.py then extract_parameters.py in sequence and
additionally exports a flat CSV for downstream analysis.

Usage:
    python pipeline.py                          # default query, 50 abstracts
    python pipeline.py --max 100 --delay 3      # 100 abstracts, 3s between calls
    python pipeline.py --query "your query"     # custom PubMed query
    python pipeline.py --skip-fetch             # re-use existing data/abstracts.json
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

from fetch_pubmed import DEFAULT_QUERY, fetch_abstracts, search_pubmed
from extract_parameters import extract_batch, HF_MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_csv(results: list, path: str) -> None:
    """Flatten nested fields (key_taxa list → pipe-separated string) for CSV."""
    if not results:
        return

    flat = []
    for r in results:
        row = dict(r)
        row["key_taxa"] = " | ".join(r.get("key_taxa") or [])
        flat.append(row)

    fieldnames = list(flat[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)


def print_summary(results: list) -> None:
    """Print a quick quality-check summary to stdout."""
    if not results:
        return

    total = len(results)
    llm_count  = sum(1 for r in results if "llm:" in r.get("extraction_method", ""))
    fallbacks  = total - llm_count
    sites      = {}
    for r in results:
        s = r.get("body_site", "unknown")
        sites[s] = sites.get(s, 0) + 1

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Total records       : {total}")
    print(f"  LLM extracted       : {llm_count}")
    print(f"  Rule-based fallback : {fallbacks}")
    print(f"\n  Body-site breakdown:")
    for site, count in sorted(sites.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {site:<20} {count:>3}  {bar}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    max_results: int,
    delay: float,
    skip_fetch: bool,
    fallback_only: bool,
    abstracts_path: str,
    extracted_path: str,
    csv_path: str,
) -> None:

    os.makedirs("data", exist_ok=True)

    # ── Step 1: Fetch abstracts ──────────────────────────────────────────
    if skip_fetch and Path(abstracts_path).exists():
        print(f"[Pipeline] Skipping fetch — loading existing {abstracts_path}")
        with open(abstracts_path, encoding="utf-8") as f:
            data = json.load(f)
        records = data["articles"]
    else:
        print("[Pipeline] Step 1/2 — Searching PubMed...")
        pmids, total = search_pubmed(query, max_results=max_results)
        time.sleep(0.4)
        records = fetch_abstracts(pmids)

        with open(abstracts_path, "w", encoding="utf-8") as f:
            json.dump({
                "query": query,
                "total_pubmed_results": total,
                "fetched": len(records),
                "articles": records,
            }, f, indent=2, ensure_ascii=False)
        print(f"[Pipeline] Abstracts saved → {abstracts_path}")

    # ── Step 2: LLM extraction ───────────────────────────────────────────
    mode_label = "rule-based fallback only" if fallback_only else f"LLM ({HF_MODEL})"
    print(f"\n[Pipeline] Step 2/2 — Extracting parameters from {len(records)} abstracts [{mode_label}]...")
    results = extract_batch(records, delay=delay, fallback_only=fallback_only)

    # Save JSON
    with open(extracted_path, "w", encoding="utf-8") as f:
        json.dump({"extracted": len(results), "results": results}, f,
                  indent=2, ensure_ascii=False)
    print(f"[Pipeline] Extracted JSON → {extracted_path}")

    # Save CSV
    save_csv(results, csv_path)
    print(f"[Pipeline] Flat CSV      → {csv_path}")

    print_summary(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PubMed Microbiome LLM Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query",       default=DEFAULT_QUERY,
                        help="PubMed search query (default: microbiome body-site 16S query)")
    parser.add_argument("--max",         type=int,   default=50,
                        help="Max abstracts to fetch (default: 50)")
    parser.add_argument("--delay",       type=float, default=2.0,
                        help="Seconds between HuggingFace API calls (default: 2)")
    parser.add_argument("--fallback-only", action="store_true",
                        help="Skip LLM; use rule-based extraction only (offline/CI mode)")
    parser.add_argument("--skip-fetch",  action="store_true",
                        help="Re-use existing data/abstracts.json (skip PubMed fetch)")
    parser.add_argument("--abstracts",   default="data/abstracts.json")
    parser.add_argument("--extracted",   default="data/extracted.json")
    parser.add_argument("--csv",         default="data/extracted.csv")
    args = parser.parse_args()

    run_pipeline(
        query          = args.query,
        max_results    = args.max,
        delay          = args.delay,
        skip_fetch     = args.skip_fetch,
        fallback_only  = args.fallback_only,
        abstracts_path = args.abstracts,
        extracted_path = args.extracted,
        csv_path       = args.csv,
    )


if __name__ == "__main__":
    main()
