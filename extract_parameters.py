"""
extract_parameters.py
---------------------
Uses a free HuggingFace Inference API model to extract structured parameters
from PubMed abstracts describing human microbiome sequencing studies.

The extraction schema mirrors health economic model parameterisation:
  body_site | sequencing_method | sample_size | study_design |
  disease_condition | key_taxa | geographic_region | main_finding

Optionally set HF_TOKEN env variable for higher rate limits:
    export HF_TOKEN=hf_your_token_here

Usage (standalone):
    python extract_parameters.py --in data/abstracts.json --out data/extracted.json
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# HuggingFace Inference API config
# ---------------------------------------------------------------------------
HF_MODEL   = "mistralai/Mistral-7B-Instruct-v0.3"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Set via environment; works without a token (free tier, rate-limited)
HF_TOKEN   = os.getenv("HF_TOKEN", "")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a biomedical data extraction assistant specialised in microbiome research. "
    "Extract structured parameters from scientific abstracts. "
    "Always return valid JSON only — no prose, no markdown, no explanation."
)

EXTRACTION_TEMPLATE = """<s>[INST] {system}

Extract the following fields from this PubMed abstract and return ONLY a JSON object:

- "body_site": primary sample site (choose from: saliva, skin, vaginal, gut_feces, urine, semen, respiratory, multiple, unknown)
- "sequencing_method": (e.g. "16S rRNA V4 amplicon", "ITS2 amplicon", "WGS", "other")
- "sample_size": integer number of human participants (null if not stated)
- "study_design": (cross-sectional | longitudinal | case-control | RCT | cohort | meta-analysis | other)
- "disease_condition": primary disease/condition studied, or "healthy_volunteers" if no disease focus
- "key_taxa": list of up to 3 microbial taxa highlighted as key findings (empty list if none mentioned)
- "geographic_region": country or region of study population (null if not stated)
- "main_finding": one concise sentence summarising the primary result
- "data_extraction_confidence": your confidence in this extraction (high | medium | low)

Abstract title: {title}

Abstract text:
{abstract}

Return ONLY the JSON object. [/INST]"""


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def build_prompt(record: Dict) -> str:
    return EXTRACTION_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        title=record.get("title", ""),
        abstract=record.get("abstract", ""),
    )


def call_hf_api(prompt: str, max_retries: int = 3) -> Optional[str]:
    """
    Call the HuggingFace Inference API. Handles model loading delays
    (HTTP 503) with exponential back-off.
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.1,        # low temp = deterministic structured output
            "return_full_text": False,
        },
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)

            if resp.status_code == 503:
                # Model is loading; wait and retry
                wait = 20 * attempt
                print(f"  [HF] Model loading, waiting {wait}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            result = resp.json()

            if isinstance(result, list) and result:
                return result[0].get("generated_text", "")
            return None

        except requests.RequestException as e:
            print(f"  [HF] Request error on attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(5 * attempt)

    return None


def parse_json_from_response(raw: str) -> Optional[Dict]:
    """
    Robustly extract JSON from LLM output — handles leading prose
    or markdown code fences the model may include despite instructions.
    """
    if not raw:
        return None

    # Try direct parse first
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Find the first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def extract_parameters_llm(record: Dict, fallback_only: bool = False) -> Dict:
    """
    Run LLM extraction on a single abstract record.
    Returns extracted parameters merged with source metadata.

    Parameters
    ----------
    fallback_only : bool
        If True, skip the LLM call and use rule-based extraction only.
        Useful for offline testing and CI validation.
    """
    if fallback_only:
        params = rule_based_fallback(record)
        params["extraction_method"] = "rule_based_fallback"
        return {
            "pmid": record.get("pmid"), "title": record.get("title"),
            "year": record.get("year"), "journal": record.get("journal"),
            "pubmed_url": record.get("pubmed_url"), **params,
        }

    prompt   = build_prompt(record)
    raw_resp = call_hf_api(prompt)
    params   = parse_json_from_response(raw_resp)

    if params is None:
        print(f"  [Extract] Could not parse JSON for PMID {record.get('pmid')}. "
              "Falling back to rule-based extraction.")
        params = rule_based_fallback(record)
        params["extraction_method"] = "rule_based_fallback"
    else:
        params["extraction_method"] = f"llm:{HF_MODEL}"

    # Merge with source metadata
    return {
        "pmid":       record.get("pmid"),
        "title":      record.get("title"),
        "year":       record.get("year"),
        "journal":    record.get("journal"),
        "pubmed_url": record.get("pubmed_url"),
        **params,
    }


# ---------------------------------------------------------------------------
# Rule-based fallback (ensures pipeline always produces output)
# ---------------------------------------------------------------------------

BODY_SITE_KEYWORDS = {
    "saliva":     ["saliva", "salivary", "oral"],
    "vaginal":    ["vaginal", "vagina", "cervicovaginal"],
    "skin":       ["skin", "cutaneous", "dermal"],
    "gut_feces":  ["feces", "fecal", "stool", "rectal", "gut", "intestinal"],
    "urine":      ["urine", "urinary", "urogenital"],
    "semen":      ["semen", "seminal"],
    "respiratory":["nasal", "lung", "airway", "respiratory", "sputum"],
}

STUDY_DESIGN_KEYWORDS = {
    "longitudinal":    ["longitudinal", "follow-up", "prospective cohort"],
    "case-control":    ["case-control", "cases and controls"],
    "RCT":             ["randomized", "randomised", "clinical trial", "rct"],
    "meta-analysis":   ["meta-analysis", "systematic review"],
    "cross-sectional": ["cross-sectional"],
}

SAMPLE_SIZE_RE = re.compile(
    r"(?:n\s*=\s*|included?\s+|enrolled?\s+|recruited?\s+|analysed?\s+|total\s+of\s+"
    r"|from\s+|of\s+|comprising\s+|spanning\s+)"
    r"(\d{2,5})\s*"
    r"(?:\w+\s+)?"   # optional adjective (e.g. "paediatric", "healthy", "hospitalized")
    r"(?:participants?|subjects?|individuals?|samples?|patients?|volunteers?|adults?|"
    r"women|men|infants?|neonates?|cases?|controls?|pregnant|healthy|preterm|HCWs?)",
    re.IGNORECASE,
)


def rule_based_fallback(record: Dict) -> Dict:
    text = (record.get("title", "") + " " + record.get("abstract", "")).lower()

    # Body site
    body_site = "unknown"
    for site, keywords in BODY_SITE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            body_site = site
            break

    # Sequencing method
    if "16s" in text or "16 s" in text:
        seq_method = "16S rRNA amplicon"
    elif "its" in text:
        seq_method = "ITS amplicon"
    elif "wgs" in text or "whole genome" in text or "shotgun" in text:
        seq_method = "WGS"
    else:
        seq_method = "amplicon (unspecified)"

    # Sample size
    sizes = SAMPLE_SIZE_RE.findall(text)
    sample_size = int(max(sizes, key=int)) if sizes else None

    # Study design
    study_design = "cross-sectional"
    for design, kws in STUDY_DESIGN_KEYWORDS.items():
        if any(kw in text for kw in kws):
            study_design = design
            break

    return {
        "body_site":                  body_site,
        "sequencing_method":          seq_method,
        "sample_size":                sample_size,
        "study_design":               study_design,
        "disease_condition":          "not_extracted",
        "key_taxa":                   [],
        "geographic_region":          None,
        "main_finding":               "Not extracted (rule-based fallback)",
        "data_extraction_confidence": "low",
    }


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_batch(records: List[Dict], delay: float = 2.0, fallback_only: bool = False) -> List[Dict]:
    """
    Extract parameters from a list of abstract records.

    `delay` — seconds to wait between API calls (respect free-tier limits).
    """
    results = []
    total   = len(records)

    for i, record in enumerate(records, 1):
        pmid = record.get("pmid", "?")
        print(f"[{i}/{total}] Extracting PMID {pmid}...")

        extracted = extract_parameters_llm(record, fallback_only=fallback_only)
        results.append(extracted)

        if i < total and not fallback_only:
            time.sleep(delay)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-assisted parameter extraction from PubMed microbiome abstracts."
    )
    parser.add_argument("--in",  dest="input",  default="data/abstracts.json",
                        help="Input JSON from fetch_pubmed.py")
    parser.add_argument("--out", dest="output", default="data/extracted.json",
                        help="Output JSON with extracted parameters")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between API calls (default: 2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N abstracts (for testing)")
    parser.add_argument("--fallback-only", action="store_true",
                        help="Skip LLM call; use rule-based extraction only (offline/CI mode)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    records = data["articles"]
    if args.limit:
        records = records[: args.limit]
        print(f"[Extract] Processing first {len(records)} abstracts (--limit applied).")

    print(f"[Extract] Starting extraction of {len(records)} abstracts using {HF_MODEL}...")
    results = extract_batch(records, delay=args.delay, fallback_only=args.fallback_only)

    output = {
        "model":     HF_MODEL,
        "extracted": len(results),
        "results":   results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[Extract] Done. Saved {len(results)} records → {args.output}")


if __name__ == "__main__":
    main()
