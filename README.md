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

## How to run this (step-by-step for beginners)

### Step 1 — Make sure you have Python installed

Open your **Terminal** (Mac: press `Cmd + Space`, type "Terminal", press Enter) and type:

```bash
python --version
```

You should see something like `Python 3.8.x` or higher. If you get an error, download Python from [python.org](https://www.python.org/downloads/).

---

### Step 2 — Download this repository

In your Terminal, run:

```bash
git clone https://github.com/meghnasw/pubmed-microbiome-llm-extractor.git
```

This creates a folder called `pubmed-microbiome-llm-extractor` wherever your Terminal is currently pointing. To see where that is, type `pwd`.

---

### Step 3 — Go into the folder

```bash
cd pubmed-microbiome-llm-extractor
```

---

### Step 4 — Install the required Python package

```bash
pip install -r requirements.txt
```

This installs `requests`, the only external library the pipeline needs. You only need to do this once.

---

### Step 5 — Run a quick offline test (no internet required)

This uses the sample abstracts already included in the `data/` folder:

```bash
python pipeline.py --skip-fetch --fallback-only
```

You should see output like this:

```
[Pipeline] Skipping fetch — loading existing data/abstracts.json
[Pipeline] Step 2/2 — Extracting parameters from 15 abstracts [rule-based fallback only]...
[1/15] Extracting PMID 38124501...
...
============================================================
EXTRACTION SUMMARY
  Total records       : 15
  Body-site breakdown:
    gut_feces            █████ (5)
    skin                 ███ (3)
    vaginal              ██ (2)
    ...
============================================================
```

Two output files are created in the `data/` folder:
- `data/extracted.json` — full structured results
- `data/extracted.csv` — flat table, openable in Excel

---

### Step 6 — Fetch real abstracts from PubMed (requires internet)

```bash
python fetch_pubmed.py --max 20 --out data/my_abstracts.json
```

This queries PubMed with the default microbiome search query and downloads 20 abstracts. You'll see: `Found 18,742 results, fetching 20.`

---

### Step 7 — Run LLM extraction on your fetched abstracts (requires internet)

```bash
python extract_parameters.py --in data/my_abstracts.json --out data/my_extracted.json --limit 5
```

This sends the first 5 abstracts to the free HuggingFace Mistral-7B model for structured extraction. **The first call may take 20–30 seconds** while the model loads on HuggingFace's servers — this is normal.

For higher rate limits, optionally create a free account at [huggingface.co](https://huggingface.co) and set your token:

```bash
export HF_TOKEN=hf_your_token_here
```

---

### Step 8 — Run the full pipeline end-to-end

```bash
python pipeline.py --max 50
```

This fetches 50 abstracts from PubMed and runs LLM extraction on all of them. Results are saved to `data/abstracts.json`, `data/extracted.json`, and `data/extracted.csv`.

---

### Custom search query

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
