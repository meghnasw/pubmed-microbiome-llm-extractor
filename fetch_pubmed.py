"""
fetch_pubmed.py
---------------
Query PubMed via NCBI E-utilities and retrieve abstracts for
human microbiome studies profiled by amplicon / 16S sequencing.

Usage (standalone):
    python fetch_pubmed.py --max 50 --out data/abstracts.json
"""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests
from typing import Tuple

# ---------------------------------------------------------------------------
# Default search query — mirrors the systematic-review inclusion criteria
# ---------------------------------------------------------------------------
DEFAULT_QUERY = (
    "microbiome "
    "AND (saliva OR semen OR skin OR vaginal OR vagina OR urine OR feces OR stool) "
    "AND human "
    "AND sequencing "
    "AND (amplicon OR 16S)"
)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# NCBI requests a contact e-mail for automated queries (not a secret)
CONTACT_EMAIL = "meghna.microbiome.extractor@research.org"


# ---------------------------------------------------------------------------
# PubMed search helpers
# ---------------------------------------------------------------------------

def search_pubmed(query: str, max_results: int = 50) -> Tuple[List[str], int]:
    """
    Run an esearch against PubMed.

    Returns
    -------
    pmids : list[str]   — PubMed IDs of matching articles
    total : int         — total number of results in PubMed for this query
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
        "email": CONTACT_EMAIL,
        "tool": "microbiome-llm-extractor",
    }
    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()["esearchresult"]
    pmids = data["idlist"]
    total = int(data["count"])

    print(f"[PubMed] Query matched {total:,} articles. Fetching {len(pmids)}.")
    return pmids, total


def fetch_abstracts(pmids: List[str]) -> List[Dict]:
    """
    Retrieve full abstract text and key metadata for a list of PMIDs.

    Handles multi-section abstracts (BACKGROUND / METHODS / RESULTS / CONCLUSIONS).
    """
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
        "email": CONTACT_EMAIL,
        "tool": "microbiome-llm-extractor",
    }
    resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    records = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el   = article.find(".//PMID")
        title_el  = article.find(".//ArticleTitle")
        year_el   = article.find(".//PubDate/Year")
        journal_el = article.find(".//Journal/Title")

        # Concatenate multi-section abstracts, preserving section labels
        abstract_sections = article.findall(".//AbstractText")
        parts = []
        for el in abstract_sections:
            label = el.get("Label")
            text  = el.text or ""
            if text:
                parts.append(f"{label}: {text}" if label else text)
        abstract_text = " ".join(parts)

        if not abstract_text:
            continue  # skip articles with no abstract

        pmid = pmid_el.text if pmid_el is not None else ""
        records.append({
            "pmid":        pmid,
            "title":       title_el.text if title_el is not None else "",
            "year":        year_el.text  if year_el  is not None else "",
            "journal":     journal_el.text if journal_el is not None else "",
            "abstract":    abstract_text,
            "pubmed_url":  f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    print(f"[PubMed] Retrieved {len(records)} abstracts with text.")
    return records


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts for microbiome LLM extraction.")
    parser.add_argument("--query",  default=DEFAULT_QUERY, help="PubMed search query")
    parser.add_argument("--max",    type=int, default=50,  help="Maximum number of abstracts to fetch")
    parser.add_argument("--out",    default="data/abstracts.json", help="Output JSON file path")
    args = parser.parse_args()

    pmids, total = search_pubmed(args.query, max_results=args.max)
    time.sleep(0.4)  # NCBI rate-limit courtesy pause
    records = fetch_abstracts(pmids)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"query": args.query, "total_pubmed_results": total,
                   "fetched": len(records), "articles": records}, f, indent=2, ensure_ascii=False)

    print(f"[PubMed] Saved {len(records)} abstracts → {args.out}")


if __name__ == "__main__":
    main()
