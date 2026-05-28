# Step 00 — Corpus & masking design (pipeline input)

This is the **starting point** of the pipeline. From the 1,746 main-track
ACL/EMNLP/NAACL 2025 papers that have both `.tex` and `.bib` on arXiv, we
extract every citation context and mask it. The masked corpus is the input that
`01_mask_and_recommend_*.py` consumes; it is distributed as data (see the repo
README), one folder per paper:

```
<MASKED_DIR>/<arxiv_id>/
├── metadata.json        # arxiv_id, doi, title, year, venue, num_contexts,
│                        #   first_arxiv_date, latest_arxiv_date
├── contexts.json        # list of citation contexts (schema below)
└── bib_entries.json     # bibkey -> {title, author, year, doi, venue}
```

## What "one citation context" is

A context is a paragraph that contains exactly one citation-bearing sentence to
be reconstructed. Each entry of `contexts.json`:

| field | meaning |
|-------|---------|
| `paragraph`            | the full source paragraph (raw LaTeX) |
| `citation_sentence`    | the sentence being masked out (the human "answer", held out from the model) |
| `cite_command`         | the raw `\cite{...}` command(s) in that sentence |
| `cite_keys`            | bibkeys cited in the masked sentence |
| `before_citation`      | the sentence immediately **before** the masked sentence |
| `after_citation`       | the sentence immediately **after** the masked sentence |
| `masked_paragraph`     | the paragraph with the citation sentence replaced by **`[CITE_HERE]`** |
| `section`              | section heading the context sits under |
| `line_number`, `paragraph_length`, `num_sentences_in_paragraph` | provenance / length bookkeeping |
| `bib_entries`          | the bib records for this context's `cite_keys` |

## Masking rules (the design)

The masking is built to make every model citation a **clean counterfactual** to
the human choice at the same slot:

1. **One sentence is removed** and replaced by the single placeholder
   `[CITE_HERE]`. The model never sees the removed human sentence nor the cited
   work — only the surrounding paragraph.
2. **±1-sentence context** is preserved (`before_citation` / `after_citation`),
   plus the section heading and the paper title/venue/year, so the local
   rhetorical context survives.
3. **Required citation count** is kept: the number of works originally cited in
   the masked sentence is given to the model, so reconstructions are
   count-controlled (tone/length differences are not confounded by retrieval
   length).
4. **Adjacent citation sentences are merged** into a single slot when they sit
   back-to-back, so one `[CITE_HERE]` corresponds to one coherent rhetorical
   move (`merge_contexts_by_sentence` in `01_mask_and_recommend_*.py`).
5. **LaTeX is stripped at prompt-build time** (`clean_paragraph_for_llm`):
   tables/figures/equations → `[TABLE]`/`[FIGURE]`/`[EQUATION]`/`[MATH]`
   placeholders, other `\cite*{...}` in the window → `[CITE]` markers (so no raw
   bibkeys leak), formatting/structural/label/url commands removed, whitespace
   collapsed, and a hard character cap that keeps `[CITE_HERE]` centered.
6. **Post-cutoff guarantee**: all 1,746 papers were released as preprints
   *after* the six models' training cutoffs, so the held-out citation cannot be
   retrieved from memory.

The result is a corpus where, for each masked slot, the human's real citation
(sentence + cited papers + the judge's reading of its intent) is known — and the
model's reconstruction is directly comparable to it along position, time, and
citation count, leaving **citation choice** as the primary axis of variation.
