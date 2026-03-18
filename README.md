# Citation LLM Workflow

This project builds a citation-context dataset from ACL preprints and runs a
two-stage LLM pipeline to study citation recommendation behavior and
motivation consistency.

## What This Repository Does

At a high level:

1. Retrieve ACL-related papers and their arXiv source packages.
2. Extract citation contexts from LaTeX and mask citation sentences.
3. Ask an LLM to recommend citations for masked contexts.
4. Ask an LLM-as-judge to classify motivation for original vs generated citation sentences.

## Requirements

- Python 3.10+
- Packages used by scripts: `pandas`, `tqdm`, `requests`, `openai`

Install quickly:

```powershell
pip install -U pandas tqdm requests openai
```

## OpenRouter / API Keys

LLM scripts (`03` and `03b`) read keys from `openrouter_keys.json` in repo root:

```json
{
  "openrouter_api_keys": [
    "sk-or-v1-KEY1",
    "sk-or-v1-KEY2"
  ]
}
```

They also support env fallbacks (`OPENROUTER_API_KEYS`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`).

## Pipeline Overview

### Step 1 - Retrieve preprints and arXiv sources

Script: `01 retreive_arxiv_preprints_acl.py`

What this file does:

- Loads the Dimensions export CSV (`data/preprints_acl_dimensions.csv`).
- Maps venue labels from DOI patterns (ACL/EMNLP/NAACL + Findings variants).
- Extracts/cleans `arxiv_id`.
- Downloads arXiv source tarballs for papers with arXiv IDs.
- Writes a new CSV with downloaded source paths.

Run:

```powershell
python "01 retreive_arxiv_preprints_acl.py"
```

Main inputs:

- `data/preprints_acl_dimensions.csv`

Main outputs:

- `data/arxiv_sources/*.tar.gz`
- `data/preprints_with_source_paths.csv`

---

### Step 2 - Extract and mask citation contexts

Script: `02 extract_mask_arxiv_preprints.py`

What this file does:

- Reads `data/preprints_with_source_paths.csv`.
- Extracts LaTeX sources per paper into `data/arxiv_extracted/{arxiv_id}/`.
- Finds main `.tex` and parses `.bib` files.
- Extracts citation contexts from paragraphs.
- Applies level-2 masking:
  - keeps one sentence before and after
  - replaces citation sentence with `[CITE_HERE]`
- Enriches each context with matching bib entries.
- Saves per-paper outputs and supports resume.

Run:

```powershell
python "02 extract_mask_arxiv_preprints.py"
```

Main outputs (per paper):

- `data/arxiv_masked/{arxiv_id}/metadata.json`
- `data/arxiv_masked/{arxiv_id}/bib_entries.json`
- `data/arxiv_masked/{arxiv_id}/contexts.json`

Important context fields in `contexts.json`:

- `citation_sentence`
- `masked_paragraph`
- `before_citation`, `after_citation`
- `cite_keys`, `bib_entries`
- `section`, `line_number`

---

### Step 3 - Generate citation recommendations

Script: `03 llm_cite_recommendation.py`

What this file does:

- Loads masked contexts from `data/arxiv_masked`.
- Filters papers (currently `year >= 2024` and has bib file).
- Deduplicates repeated raw contexts into unique citation-sentence contexts.
- Handles multi-citation cases:
  - computes required citation count from `cite_keys`
  - asks model to return exactly that many recommended papers
  - keeps one replacement citation sentence
- Calls OpenRouter model (`deepseek/deepseek-chat-v3.1`) with:
  - multiple keys (round-robin)
  - parallel workers
  - retry + backoff
- Validates response schema/count and saves per-paper responses.

Run:

```powershell
python "03 llm_cite_recommendation.py"
```

Main output:

- `data/arxiv_llm_responses/{arxiv_id}/responses.json`

Important result fields:

- `citation_sentence_original`
- `num_citations_required`, `num_citations_returned`
- `llm_response.recommended_papers[]`
- `llm_response.citation_sentence`
- `llm_response.motivation`, `confidence`

---

### Step 4 - Judge motivation (original vs generated)

Script: `03b llm_categorize_origin_citation.py`

What this file does:

- Reads Step 3 output (`data/arxiv_llm_responses`).
- For each context, runs LLM-as-judge on:
  - original citation sentence
  - LLM-filled citation sentence
- Uses the motivation taxonomy:
  - `supporting`
  - `contrasting`
  - `mentioning`
- Uses same OpenRouter multi-key + parallel + retry setup.
- Saves comparison-friendly outputs per paper.

Run:

```powershell
python "03b llm_categorize_origin_citation.py"
```

Main output:

- `data/arxiv_original_llm_responses/{arxiv_id}/responses.json`

Important result fields:

- `motivation_script03_self`
- `motivation_llm_as_judge_original`
- `motivation_llm_as_judge_filled`

---

## Optional Analysis

Script: `04 criticism_analysis_dictionary.py`

Use this after Step 3/4 to aggregate and analyze disagreement patterns.

## Suggested Run Order

```powershell
python "01 retreive_arxiv_preprints_acl.py"
python "02 extract_mask_arxiv_preprints.py"
python "03 llm_cite_recommendation.py"
python "03b llm_categorize_origin_citation.py"
```
