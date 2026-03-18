# Citation LLM Workflow (4 Steps)

This project runs a 4-step pipeline to study citation recommendation behavior.

## Step 1 - Retrieve ACL preprints and arXiv sources

Script: `01 retreive_arxiv_preprints_acl.py`

Run:

```powershell
python "01 retreive_arxiv_preprints_acl.py"
```

Output:
- `data/preprints_with_source_paths.csv`

## Step 2 - Extract and mask citation contexts

Script: `02 extract_mask_arxiv_preprints.py`

Run:

```powershell
python "02 extract_mask_arxiv_preprints.py"
```

Output (per paper):
- `data/arxiv_masked/{arxiv_id}/metadata.json`
- `data/arxiv_masked/{arxiv_id}/bib_entries.json`
- `data/arxiv_masked/{arxiv_id}/contexts.json`

## Step 3 - Generate citation recommendations

Script: `03 llm_cite_recommendation.py`

Run:

```powershell
python "03 llm_cite_recommendation.py"
```

Output:
- `data/arxiv_llm_responses/{arxiv_id}/responses.json`

## Step 4 - Judge motivation for original vs generated citations

Script: `03b llm_categorize_origin_citation.py`

Run:

```powershell
python "03b llm_categorize_origin_citation.py"
```

Output:
- `data/arxiv_original_llm_responses/{arxiv_id}/responses.json`
