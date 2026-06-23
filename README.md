# PubMed Microbiome LLM Extractor

**AI-assisted systematic review pipeline for human microbiome studies**

Automatically searches PubMed for human microbiome sequencing studies, then uses a large language model (LLM) to extract structured parameters from each abstract — body site, study design, sample size, key taxa, and more.

This mirrors the core challenge in **health economic modeling pipelines**: systematically extracting structured model parameters (states, transitions, costs, utilities) from heterogeneous scientific literature at scale.

---

## What it does

```
PubMed search query
      ↓
  NCBI E-utilities API  →  raw abstracts (JSON)
      ↓
  LLM parameter extraction  →  structured records (JSON + CSV)
      ↓
  Rule-based validation fallback (always produces output)
```

### Extracted fields

| Field | Description |
|---|---|
| `body_site` | Saliva, skin, vaginal, gut/feces, urine, semen, respiratory |
| `sequencing_method` | 16S rRNA amplicon, WGS, ITS, etc. |
| `sample_size` | Number of human participants |
| `study_design` | Cross-sectional, longitudinal, case-control, RCT, meta-analysis |
| `disease_condition` | Primary disease or healthy volunteers |
| `key_taxa` | Up to 3 microbial taxa highlighted as key findings |
| `geographic_region` | Country or region of study population |
| `main_finding` | One-sentence summary of the primary result |
| `data_extraction_confidence` | High / medium / low |
| `extraction_method` | LLM model used, or `rule_based_fallback` |

---

## Default search query

```
microbiome
AND (saliva OR semen OR skin OR vaginal OR vagina OR urine OR feces OR stool)
AND human
AND sequencing
AND (amplicon OR 16S)
```

This query retrieves **~18,700 PubMed articles** (as of June 2026). The pipeline fetches the top N by relevance.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

No API key required. For higher HuggingFace rate limits, optionally set:

```bash
export HF_TOKEN=hf_your_token_here
```

### 2. Run the full pipeline

```bash
# Fetch 50 abstracts and extract parameters
python pipeline.py --max 50

# Outputs:
#   data/abstracts.json    — raw PubMed abstracts
#   data/extracted.json    — structured extraction results
#   data/extracted.csv     — flat CSV for downstream analysis
```

### 3. Run steps individually

```bash
# Step 1: fetch abstracts only
python fetch_pubmed.py --max 100 --out data/abstracts.json

# Step 2: extract from existing abstracts (skip re-fetching)
python extract_parameters.py --in data/abstracts.json --out data/extracted.json

# Step 2 alt: use pipeline with --skip-fetch
python pipeline.py --skip-fetch
```

### 4. Custom query

```bash
python pipeline.py --query "gut microbiome AND (RSV OR influenza OR COVID) AND vaccine" --max 30
```

---

## Sample output

From `data/extracted.csv` (15 abstracts, rule-based extraction):

| PMID | body_site | sample_size | study_design | disease_condition | sequencing_method |
|---|---|---|---|---|---|
| 38124501 | vaginal | 312 | cross-sectional | not_extracted | 16S rRNA amplicon |
| 37891234 | gut_feces | 1247 | case-control | not_extracted | 16S rRNA amplicon |
| 38203847 | saliva | 180 | cross-sectional | not_extracted | 16S rRNA amplicon |
| 37654321 | skin | 94 | longitudinal | not_extracted | 16S rRNA amplicon |
| 38056789 | urine | 225 | cross-sectional | not_extracted | 16S rRNA amplicon |
| 37812345 | semen | 156 | cross-sectional | not_extracted | 16S rRNA amplicon |

> **Note:** `disease_condition` and `main_finding` fields are populated by the LLM backend (HuggingFace). The rule-based fallback extracts body site, sample size, sequencing method, and study design from text patterns. Run with the HuggingFace API for full extraction.

---

## Architecture

### `fetch_pubmed.py`
- Queries NCBI E-utilities `esearch` + `efetch` endpoints
- Handles multi-section abstracts (BACKGROUND / METHODS / RESULTS / CONCLUSIONS)
- Rate-limit compliant (NCBI tool + email registration)

### `extract_parameters.py`
- Calls **HuggingFace Inference API** (`mistralai/Mistral-7B-Instruct-v0.3`)
- Structured JSON prompt with schema enforcement
- Handles 503 model-loading delays with exponential back-off
- **Rule-based fallback** (regex + keyword matching) ensures the pipeline always produces output even without API access — useful for CI/CD validation

### `pipeline.py`
- Orchestrates both steps end-to-end
- Exports JSON + flat CSV
- Prints a body-site frequency summary on completion

---

## Extending the pipeline

### Add a new extraction field

In `extract_parameters.py`, add the field to `EXTRACTION_TEMPLATE` and `rule_based_fallback()`.

### Use OpenAI instead of HuggingFace

Replace `HF_API_URL` and `call_hf_api()` with an OpenAI chat completion call:

```python
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
)
return response.choices[0].message.content
```

### Adapt for health economic model extraction

Change the extraction schema in `EXTRACTION_TEMPLATE` to target health economic parameters:

```
- "model_type": (Markov | decision tree | microsimulation | dynamic transmission | other)
- "health_states": list of model health states
- "transition_parameters": key transition probabilities with source
- "cost_parameters": costs with perspective (payer | societal) and currency
- "utility_values": EQ-5D / SF-6D / other utility weights
- "time_horizon": model time horizon
- "discount_rate": applied discount rate
```

This is the direct extension of this pipeline toward infectious disease health economic modeling.

---

## Requirements

- Python 3.9+
- `requests` (see `requirements.txt`)
- Internet access (PubMed E-utilities + HuggingFace Inference API)
- HF_TOKEN env variable (optional; increases HuggingFace rate limits)

---

## Author

**Dr. Meghna Swayambhu**  
Computational Scientist · AI/ML Pipelines · Infectious Disease Data Science  
[LinkedIn](#) · [GitHub](#)

---

## Related work

This pipeline is part of a broader research focus on applying LLM-based information extraction to systematic review automation in biomedical and health economic modeling contexts.
