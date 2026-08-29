# Citing Less Critically: LLMs Reshape the Rhetoric and Reach of Scientific Citation

Code for a masked-citation study that compares **human** and **LLM-generated**
citation behavior across six popular LLMs and 1,746 top-NLP-conference papers
(ACL / EMNLP / NAACL Main 2025; 63k+ citation contexts, 132k+ citation slots).

In science, citations carry rhetorical **intent** — scholars cite prior work
positively (*supporting*), negatively (*contrasting*), or neutrally
(*mentioning*). We cast citation as a **reconstruction task**: each citation
sentence is masked, an LLM regenerates it (real papers + a fill sentence + a
self-reported intent), and the result is a counterfactual corpus
**position-aligned** to the real human choice at the same slot. An LLM-as-a-judge
then labels every original and reconstructed sentence's intent, and a 20.3M-edge
coauthorship network measures the social distance between citing and cited
authors.

Three patterns emerge, organized as the paper's three research questions:

| RQ | Question | Headline finding |
|----|----------|------------------|
| **RQ1** | Do LLMs reproduce the human distribution of citation **intents**? | LLMs cite **less critically** — they under-produce *contrasting* citations and rewrite ~half of genuinely critical contexts as supporting. |
| **RQ2** | Does intent modulate the **bias** in cited-paper attributes? | LLMs over-cite popular/older work, and the human–LLM gap **peaks at a different intent for each attribute**: popularity at *supporting*, recency at *contrasting*, team size at *mentioning*. |
| **RQ3** | Does coauthorship **proximity** shape LLM citation as it does for humans? | Humans cite within their close collaboration neighborhood (especially when supporting); **no LLM does** — they draw on more socially distant authors. |

---

## Repository layout

```
.
├── README.md                  # this file
├── requirements.txt           # pip dependencies
├── environment.yml            # conda environment (alternative to pip)
├── config.example.yml         # config template → copy to config.yml (git-ignored)
├── api_keys.example.json      # combined API-key template (both providers) → copy to api_keys.json
├── .gitignore                 # excludes secrets, data, and generated figures
│
├── assets/                    # small utility lookup tables (committed)
│   └── country_codes.csv      # ISO 3166-1 alpha-3/alpha-2 → country name
│
├── pipeline/                  # data generation: raw papers → analysis-ready CSVs
│   ├── 00_masked_corpus.md                     # masking design + masked-corpus schema (input)
│   ├── 01_mask_and_recommend.py                # LLM reconstructs each masked citation (OpenRouter or SiliconFlow)
│   ├── 02_judge_intent.py                      # LLM-as-judge intent labels (two-phase)
│   ├── 03_match_dimensions.py                  # match references to Dimensions (DOI → title)
│   ├── 04_build_author_dyads.py                # per-slot first/last-author dyads + metadata
│   ├── 05_build_coauthor_graph.py              # 2015–2024 coauthorship edge list
│   └── 06_compute_distances.py                 # BFS shortest-path dyad distances
│
└── analysis/                  # figure generation, organized by paper section
    ├── common.py              # shared loaders, palette, style, stats helpers
    ├── rq1_intent_shift.py            # §3  Intent shift from human citations
    ├── rq2_bias_amplification.py      # §4  Bias amplified by LLM intent
    └── rq3_social_proximity.py        # §5  Citation and social proximity
```

> The scripts in `pipeline/` are cleaned copies of the original working scripts;
> the originals are left untouched. They are renumbered `00–06` to read as a
> linear pipeline (the originals used a `03 / 04 / 05a–d / 06 / 06b` scheme).
>
> **Datasets are distributed separately** from this code. Per the Dimensions
> terms of service we release only code and aggregate, de-identified outputs.
> See [§ Data](#data) for what the analysis scripts expect.

---

## The pipeline (every process, end to end)

The pipeline runs in two halves. **Stage A (00–02)** turns raw papers into
intent-labeled, position-aligned human/LLM citation pairs. **Stage B (03–06)**
grounds every reference in Dimensions and places its authors on a coauthorship
network. Stage B uses Google BigQuery against an institutional Dimensions
subscription; tables are cached, so re-runs are cheap (set the `SKIP_BQ_*` env
flags documented in each script's header).

```
  papers (.tex/.bib on arXiv)
        │  00  mask each citation → [CITE_HERE], keep ±1-sentence context + required count
        ▼
  masked corpus  ──01──►  LLM reconstruction (real papers + fill sentence + self-intent)
        │                        │
        │  02  LLM-as-judge intent on BOTH the human original and the LLM-filled sentence
        ▼                        ▼
  intent-labeled, position-aligned human vs. LLM citation pairs
        │  03  match references to Dimensions (DOI → normalized title)
        │  04  resolve focal & cited first/last authors → researcher IDs + metadata
        │  05  build the 2015–2024 coauthorship graph (2.1M nodes, 20.3M edges)
        │  06  BFS shortest-path distance for the 4 author-role dyads
        ▼
  *_with_distances.csv  ──►  analysis/  (RQ1 / RQ2 / RQ3 figures)
```

### 00 — Corpus & masking design  ·  [`pipeline/00_masked_corpus.md`](pipeline/00_masked_corpus.md)

The **start of everything.** From the 1,746 main-track ACL/EMNLP/NAACL 2025
papers with `.tex` + `.bib` on arXiv, every citation context is extracted and
the citation sentence is replaced by a single `[CITE_HERE]` placeholder. The
masked corpus (one folder per paper: `metadata.json`, `contexts.json`,
`bib_entries.json`) is the pipeline input. Masking is designed so each model
citation is a **clean counterfactual** to the human choice at the same slot:

- **One sentence removed** → `[CITE_HERE]`; the model never sees the human sentence or the cited work.
- **±1-sentence context** preserved, plus section heading and paper title/venue/year.
- **Required citation count** supplied → count-controlled reconstruction (tone/length not confounded by retrieval length).
- **Adjacent citation sentences merged** into one slot (one `[CITE_HERE]` = one rhetorical move).
- **LaTeX stripped at prompt time** (tables/figures/equations → placeholders; other `\cite*{}` → `[CITE]` so no bibkeys leak).
- **Post-cutoff**: all papers released after the six models' training cutoffs, so the held-out citation can't be retrieved from memory.

See the doc for the full `contexts.json` schema and rules.

### 01 — Masking & reconstruction  ·  `pipeline/01_mask_and_recommend.py`

For each masked slot the model receives the cleaned masked paragraph, section,
title/venue/year, and the required count, and returns that many **real** papers,
a reconstructed sentence, and a **self-reported intent + confidence**, as strict
JSON. Temperature = 0 for all models. One script serves both OpenAI-compatible
providers — pick with `PROVIDER=openrouter` (default) or `PROVIDER=siliconflow`;
keys load from the **external `api_keys.json` you supply** (provider-specific
field) — no keys are committed. Resume-safe (skips papers already done).

Six source models: **GPT-5.1, Claude-3.5-Haiku, Gemini-2.0-Flash,
DeepSeek-V3.2, Llama-4-Maverick, Qwen2.5-72B-Instruct**.

<details><summary><b>Generation prompt</b> (system + user template)</summary>

```
SYSTEM
You reconstruct removed citations from NLP papers. Given a paragraph where
[CITE_HERE] replaces one removed citation-bearing sentence, infer the cited
paper(s) and the replacement sentence using the surrounding context.
Return ONLY a valid JSON object:
{ "citation_count": <int>,
  "recommended_papers": [ {"title","authors","year","venue","doi" | null} ],
  "citation_sentence": "<one sentence replacing [CITE_HERE]>",
  "motivation": "supporting" | "contrasting" | "mentioning",
  "confidence": "high" | "medium" | "low" }
Rules: recommended_papers length must equal citation_count, which must equal the
required count; each item is one distinct work; prefer real, specific papers.

USER
Paper: "<title>"   Venue/year: <venue>, <year>   Section: <section>
<masked paragraph with [CITE_HERE], LaTeX clutter stripped>
Required citation count: <N>
```
</details>

### 02 — Intent labeling, LLM-as-judge  ·  `pipeline/02_judge_intent.py`

An independent judge labels each sentence *supporting / contrasting / mentioning*
(+ confidence), seeing the title, section, and ±1-sentence window but **never the
cited-paper titles** — so the label reflects the *rhetorical move*, not which
work is cited. The **same prompt** scores the human original and every model
reconstruction, so any label-distribution difference is attributable to the
sentence, not the procedure. Two phases:

- **Phase A** judges the **original** human sentence — once per judge, reusable across all source models.
- **Phase B** judges each **LLM-filled** sentence — once per source×judge pair — and embeds the matching Phase-A label, so each record is self-contained. The judge never sees both versions of a slot (no anchoring).

Primary judge **Gemini-3-Flash-Preview** (temperature 0); robustness replication
**DeepSeek-V4-Flash**. Idempotent (per-paper + per-context resume).

<details><summary><b>Judge prompt</b> (identical across Phase A and B)</summary>

```
SYSTEM
You classify citations in NLP papers as supporting, contrasting, or mentioning,
based on a citation sentence and its surrounding context.
- supporting: cited work provides evidence/methods/findings aligned with the citing paper.
- contrasting: competing approach, contradicting finding, or baseline improved upon/disagreed with.
- mentioning: background, definitions, or general acknowledgment with no clear support/contrast.
Confidence: high (explicit) / medium (strongly implied) / low (ambiguous).
Return ONLY: {"motivation": "...", "confidence": "..."}

USER
Paper: "<title>"   Section: <section>
Sentence before: <cleaned before-window or "(beginning of paragraph)">
Citation sentence: <original (Phase A) OR model-filled (Phase B), LaTeX stripped>
Sentence after:  <cleaned after-window or "(end of paragraph)">
```
The before/after window is capped at ~500 chars each, the sentence at 800; cited
titles are deliberately omitted.
</details>

### 03 — Matching citations to papers  ·  `pipeline/03_match_dimensions.py`

Each human- and LLM-generated reference is linked to a **Dimensions** record:
**DOI first**, then exact normalized-title fallback (HTML-decode → LaTeX-strip →
NFKD ASCII-fold → lowercase → drop articles → collapse spaces). Matched
references inherit Dimensions metadata (team size, recency, citation count,
venue, type); unmatched references are dropped; author names are *not* matched.
Human references match at 86.7%; LLMs vary 39.5–81.9% — the gap itself is a
finding (hallucinated/malformed references).

### 04 — Author dyads  ·  `pipeline/04_build_author_dyads.py`

For every matched slot, resolve the **first and last author** of both the focal
and cited papers to Dimensions researcher IDs, and attach longitudinal
researcher metadata (career age, productivity, citations received before 2025,
first-affiliation country). One row per citation slot.

### 05 — Coauthorship network construction  ·  `pipeline/05_build_coauthor_graph.py`

The **graph construction design** that RQ3 depends on. We identify the first/last
author of every focal and cited paper (89k+ unique authors across 104k+ matched
papers) and build a **10-year (2015–2024) coauthorship graph**:

- **Node** = a disambiguated Dimensions researcher ID.
- **Edge** = an undirected link between **every pair of coauthors within any
  publication that contains one of our focal/cited authors** (the within-pub
  "G2" rule). Edge weight = number of joint 2015–2024 publications.
- **Year window** `[2015, 2025)` — a pre-2015 collaborator who hasn't co-published
  since 2015 is *not* an edge, so distances reflect the *recent* collaboration
  structure.
- **Team-size cap (≤ 25)**: publications with more than 25 disambiguated authors
  are dropped, to prevent the C(N,2) edge explosion from mega-team consortium
  papers.
- **Canonical, deduplicated** edge list (`Source < Target`, loadable directly as
  an undirected weighted graph).

This publication-induced graph retains the 1-hop coauthor neighborhoods around
focal and cited authors — the local collaboration structure most relevant to the
observed citation pairs — while staying tractable. Result: **2.1M+ nodes, 20.3M+
edges.**

### 06 — Dyad distances  ·  `pipeline/06_compute_distances.py`

For each slot, run BFS shortest-path on the graph for the four author-role dyads
(focal-first/last ↔ cited-first/last) and append four `dist_*` columns:
`0` = self-cite (same person), `1` = direct coauthor, larger = more distant,
`-1` = both present but disconnected, `NaN` = a researcher ID was missing. Each
citation is summarized by the **mean reachable distance ⟨d⟩** over its (up to
four) dyads. The resulting `*_with_distances.csv` files are the
**analysis-ready inputs** for everything in `analysis/`.

---

## Analysis (organized by the paper's structure)

Run any script standalone; each prints the data points in scope, then saves its
figures into `figures/` (override with `CITATION_FIG_DIR`). Every figure uses a
consistent style — de-framed axes, dashed horizontal grid, TrueType-embedded
fonts for vector-editable PDF/SVG.

```bash
cd analysis
export CITATION_DATA_DIR=/path/to/author_dyads_judge_gemini   # the *_with_distances.csv files
python rq1_intent_shift.py
python rq2_bias_amplification.py
python rq3_social_proximity.py
```

### §3 / RQ1 — Intent shift  ·  `rq1_intent_shift.py`
| figure (PNG + PDF) | what it shows |
|--------|---------------|
| `motivation_label_preservation`            | Of the citations the judge labeled on the human sentence, how often the same label survives on the LLM-filled sentence. Contrasting is least preserved (34.6–50.6%) for every model. |
| `motivation_per_paper_aggregate`           | Per-paper intent rate (one observation per paper, 95% CI) vs. the human dashed baseline. Five of six models inflate *supporting*; all six suppress *contrasting*. |
| `step3_section_supporting_curve_normalized` | Per-section *supporting* rate ÷ each file's own mean (1.0 = its average), isolating where each model over/under-supports. |
| `step3_section_contrasting_curve_normalized`| Same, for *contrasting*: LLMs hedge in argumentative sections (esp. Discussion), reserving criticism for Limitations/Experiments. |

### §4 / RQ2 — Bias amplified by intent  ·  `rq2_bias_amplification.py`
| figure (PNG + PDF) | what it shows |
|--------|---------------|
| `popularity_gap_by_intent_perpaper` | Per-paper geometric-mean ratio LLM/human of cited **citation count**, by intent. Peaks at *supporting* (DeepSeek 4.45×, GPT-5.1 3.59×). |
| `recency_gap_by_intent_perpaper`    | Per-paper mean difference LLM−human in **recency** (years older). Peaks at *contrasting* (up to +3.3 yr). |
| `teamsize_gap_by_intent`            | Citation-level ratio LLM/human of cited **team size**. Most extreme at *mentioning* (down to 0.30×). |

The most-deviant intent per model is outlined and starred (test against the
reference intent).

### §5 / RQ3 — Citation and social proximity  ·  `rq3_social_proximity.py`
| figure (PNG + PDF + SVG) | what it shows |
|--------|---------------|
| `distance_kde_per_context_mean`               | KDE of per-context mean reachable coauthor distance ⟨d⟩, one curve per file. Humans peak closer (⟨d⟩=3.40); LLMs sit right (3.65–3.89). |
| `errorbar_distance_combined_per_context__mean`| ⟨d⟩ by intent, all files on one axis with ANOVA + pairwise Welch brackets. Humans show a sharp supporting<contrasting/mentioning gradient; LLMs cluster flat. |
| `near_network_d_le_1_combined`                | Per-paper in-network rate (d ≤ 1: self-cite or direct coauthor) by intent. Humans 7–10% (elevated for supporting, p<0.01); every LLM ≤1.6% with no gradient. |

---

## Setup

Pick one:

```bash
# conda (recommended — pulls networkit/pyarrow cleanly)
conda env create -f environment.yml && conda activate citation-intent

# or pip
pip install -r requirements.txt
```

- The analysis scripts (`analysis/`) need only `numpy`, `pandas`, `scipy`, `matplotlib`.
- The pipeline scripts (`pipeline/`) additionally need `openai`, `tqdm`,
  `networkit`, `pyarrow`, `google-cloud-bigquery`, and `pyyaml`.

## Configuration & credentials

**No secrets are committed.** All credential- and account-specific values are
externalized; copy the templates and fill in your own, or export the equivalent
environment variables.

| you need | what to do |
|----------|------------|
| **LLM API keys** (steps 01–02) | `cp api_keys.example.json api_keys.json` and paste your OpenRouter and/or SiliconFlow key(s) under their fields (one combined file; each script reads only its provider's field — point `KEYS_FILE=api_keys.json`). Or set `OPENROUTER_API_KEYS` / `SILICONFLOW_API_KEYS`. The real `*keys.json` is git-ignored. |
| **BigQuery project / dataset** (steps 03–06) | `export BQ_PROJECT=...` and `export BQ_DATASET=...` (your private working dataset). Defaults are the placeholders `your-gcp-project` / `your_dataset`. Authenticate with `gcloud auth application-default login`. |
| **Table names** | Documented in `config.example.yml` under `bigquery.tables`; each script builds `"<project>.<dataset>.<table>"`, so renaming the dataset redirects everything. The Dimensions vendor source tables (`dimensions-ai.data_analytics.*`) are the same for every subscriber. |
| **Where the data lives** | `export CITATION_DATA_DIR=...` (the `*_with_distances.csv` dir) and optionally `CITATION_FIG_DIR=...` for outputs. |

`config.example.yml` is the single overview of every knob (project, dataset,
table names, provider URLs, model lists, paths). Copy it to `config.yml`
(git-ignored) to keep your real values out of version control.

## Data availability

This repo ships **code only**; data is hosted separately and git-ignored.

**To replicate the analysis**, download the **analysis-ready replication
package** — the already masked → reconstructed → intent-judged →
Dimensions-matched → distance-augmented per-source CSVs (output of pipeline
steps 00–06) — and unpack it into `data/`. See **[`data/README.md`](data/README.md)**
for the download links and exact layout. In short:

```bash
cd data && unzip citation_intent_replication_data.zip     # → data/author_dyads_judge_gemini/
cd ../analysis && export CITATION_DATA_DIR=../data/author_dyads_judge_gemini
python rq1_intent_shift.py    # + rq2_…, rq3_…
```

The package is hosted on **Zenodo** (citable DOI; primary), mirrored on
**Hugging Face Datasets** and **Google Drive** — fill the real links into
`data/README.md`.

| kind | where |
|------|-------|
| Analysis-ready replication CSVs (`*_with_distances.csv`, per judge) | **Zenodo / HF / Drive** → unpack to `data/author_dyads_judge_{gemini,deepseek}/`; point `CITATION_DATA_DIR` at it. |
| Bulk intermediate artifacts (masked corpus, parquet edge lists, raw judge JSON) | same hosts, separate archives; only needed to re-run the pipeline. |
| Small utility / reference tables (e.g. `assets/country_codes.csv`) | **committed** in `assets/`. |
| Generated figures (`*.png/.pdf/.svg`) and any `*.csv`/`*.html` | git-ignored; regenerate locally by running the analysis scripts. |

Per the Dimensions terms of service, only the aggregate, de-identified
analysis-ready form is redistributed — not the raw Dimensions records or the
full coauthorship network.

The analysis scripts read **`<CITATION_DATA_DIR>/*_with_distances.csv`** — one
file per source (ORIGINAL + the six LLMs), produced by
`pipeline/06_compute_distances.py`. Each CSV is one row per citation slot
(`focal_arxiv_id` × `context_index` × `position`); the columns the analysis uses:

| group | columns |
|-------|---------|
| intent | `motivation_judge_original` (human), `motivation_judge_filled` (LLM), `motivation_self` |
| context | `section`, `focal_venue`, `focal_year`, `focal_arxiv_id`, `context_index`, `position` |
| cited attrs | `cited_dim_id`, `cited_year`, `cited_citations_count`, `cited_teamsize` |
| distances | `dist_focal_first_to_cited_first`, `dist_focal_first_to_cited_last`, `dist_focal_last_to_cited_first`, `dist_focal_last_to_cited_last` |

## Reproducibility & robustness

All findings replicate under a **second independent judge** (DeepSeek-V4-Flash;
re-run the analysis with `CITATION_DATA_DIR` pointed at the DeepSeek-judge CSVs)
and on the **shared-context intersection** — the 12,556 contexts (1,695 of 1,746
papers) where every source resolves at least one Dimensions-matched cite — ruling
out judge choice and matching-coverage variation as the source of the effects.
