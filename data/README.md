# Data — replication package

The code in this repo ships **without** the data files (they're git-ignored).
To **replicate the analysis** (the RQ1/RQ2/RQ3 figures), download the
**analysis-ready, de-identified** release and unpack it here.

This release is the already **masked → reconstructed → intent-judged → Dimensions-matched → distance-augmented** data — i.e. the output of pipeline steps 00–06 — so you can reproduce every figure without re-running the LLM, BigQuery, or graph steps.

## Download

> Replace the placeholder links below with the real ones for your release.

| mirror | link |
|--------|------|
| **Zenodo** (citable DOI — primary) | `https://doi.org/10.5281/zenodo.XXXXXXX` |
| **Hugging Face Datasets** | `https://huggingface.co/datasets/<org>/citation-intent-replication` |
| **Google Drive** (mirror) | `https://drive.google.com/drive/folders/XXXXXXXX` |

```bash
# from the repo root, after downloading the archive into data/
cd data
unzip citation_intent_replication_data.zip      # or: tar -xzf ...
```

## Expected layout after unpacking

The analysis scripts read whatever `CITATION_DATA_DIR` points at; by default that
is `data/author_dyads_judge_gemini/`. Keep the file names exactly as below — the
loaders discover sources by the `*_with_distances.csv` suffix.

```
data/
└── author_dyads_judge_gemini/                 # primary judge: Gemini-3-Flash-Preview
    ├── original_dyads_with_distances.csv                         # ORIGINAL (human)
    ├── llm_generated__openai_gpt-5.1-chat_with_distances.csv
    ├── llm_generated__anthropic_claude-3.5-haiku_with_distances.csv
    ├── llm_generated__google_gemini-2.0-flash-001_with_distances.csv
    ├── llm_generated__deepseek-ai_DeepSeek-V3.2_with_distances.csv
    ├── llm_generated__meta-llama_llama-4-maverick_with_distances.csv
    └── llm_generated__Qwen_Qwen2.5-72B-Instruct_with_distances.csv
```

The robustness-check judge (DeepSeek-V4-Flash) ships as a parallel folder with
identical file names:

```
data/
└── author_dyads_judge_deepseek/               # same 7 files, second judge
    └── ...
```

## Run the analysis against it

```bash
cd ../analysis
export CITATION_DATA_DIR=../data/author_dyads_judge_gemini      # or .../judge_deepseek
python rq1_intent_shift.py
python rq2_bias_amplification.py
python rq3_social_proximity.py
```

## What's in each CSV

One row per citation slot (`focal_arxiv_id` × `context_index` × `position`).
The columns the analysis uses: intent labels (`motivation_judge_original`,
`motivation_judge_filled`, `motivation_self`), context (`section`,
`focal_venue`, `focal_year`, …), cited-paper attributes (`cited_dim_id`,
`cited_year`, `cited_citations_count`, `cited_teamsize`), and the four BFS
coauthor distances (`dist_focal_{first,last}_to_cited_{first,last}`).

> Per the Dimensions terms of service, only this aggregate, de-identified
> analysis-ready form is redistributed — not the raw Dimensions records or the
> full coauthorship network. Re-deriving these CSVs from scratch requires your
> own Dimensions/BigQuery access (pipeline steps 03–06).
