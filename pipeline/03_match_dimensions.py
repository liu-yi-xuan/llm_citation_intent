"""
Step 03 — Match every citation (originals + LLM-generated) in the Step-02
Phase-B outputs to dimensions-ai.data_analytics.publications, via BigQuery.

Symmetric DOI-first → title-fallback matching for BOTH originals AND
LLM-generated citations.

Output: one JSON file per source model under
  data/dimensions_dointitle_matched_judge_gemini/

BigQuery side: stages a single table
  <project>.<dataset>.citations_to_match
JOINs once against dimensions-ai.data_analytics.publications, writes
  <project>.<dataset>.citations_matched_dointitle
Default behavior: overwrite both tables and all output JSON files.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
import unicodedata
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery


# ============================================================
# CONFIG
# ============================================================

JUDGE_BASE_DIR        = "data/arxiv_llm_judge_gemini"
JUDGE_MODEL_NAME      = "google/gemini-3-flash-preview"
JUDGE_SLUG            = "google_gemini-3-flash-preview"

PREPRINTS_CSV         = "data/metadata/preprints_acl_dimensions.csv"
PREPRINTS_SOURCE_CSV  = "data/metadata/preprints_with_source_paths.csv"

OUT_DIR               = "data/dimensions_dointitle_matched_judge_gemini"

# Configure via env (or config.yml — see config.example.yml). Anonymized defaults.
BQ_PROJECT            = os.environ.get("BQ_PROJECT", "your-gcp-project")
BQ_DATASET            = os.environ.get("BQ_DATASET", "your_dataset")
BQ_INPUT_TABLE        = f"{BQ_PROJECT}.{BQ_DATASET}.citations_to_match"
BQ_OUTPUT_TABLE       = f"{BQ_PROJECT}.{BQ_DATASET}.citations_matched_dointitle"
BQ_FOCAL_INPUT_TABLE  = f"{BQ_PROJECT}.{BQ_DATASET}.focal_papers_to_match"
BQ_FOCAL_OUTPUT_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.focal_papers_matched"
DIMENSIONS_TABLE      = "dimensions-ai.data_analytics.publications"

# Set BQ_LOCATION if your dataset is in a non-US region.
BQ_LOCATION           = "US"

PRINT_EVERY           = 500


# ============================================================
# NORMALIZATION HELPERS — mirrored in SQL below
# ============================================================

def normalize_doi(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    v = value.strip().lower()
    v = re.sub(r"^https?://(dx\.)?doi\.org/", "", v)
    v = re.sub(r"^doi:\s*", "", v)
    return v.strip()


_TITLE_STOPWORDS = re.compile(r"\b(?:the|a|an)\b")


def normalize_title(value: Any) -> str:
    """Normalize a title for cross-DB exact matching.

    Pipeline:
      1. HTML-decode (&amp; -> &)
      2. Strip LaTeX commands (\\textbf{X} -> X, \\alpha -> '')
      3. NFKD-decompose + drop combining marks (folds accents: é -> e, ñ -> n)
      4. Lowercase
      5. Replace any remaining non-[a-z0-9 ] character with a space
      6. Drop the short stopwords {the, a, an}
      7. Collapse whitespace
    Mirrored on the BigQuery side via NORMALIZE(.., NFKD) + REGEXP_REPLACE."""
    if not isinstance(value, str):
        return ""
    # 1. HTML entities
    v = html.unescape(value)
    # 2. LaTeX-stripping
    v = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", v)
    v = re.sub(r"\\[a-zA-Z]+", "", v)
    v = re.sub(r"[{}]", "", v)
    # 3. NFKD decomposition + drop combining marks (\p{Mn})
    v = unicodedata.normalize("NFKD", v)
    v = "".join(ch for ch in v if not unicodedata.combining(ch))
    # 4. Lowercase
    v = v.lower()
    # 5. Non-alnum (post-fold) -> space
    v = re.sub(r"[^a-z0-9\s]", " ", v)
    # 6. Drop short stopwords
    v = _TITLE_STOPWORDS.sub(" ", v)
    # 7. Collapse whitespace
    v = re.sub(r"\s+", " ", v).strip()
    return v


def bare_arxiv_id(v: Any) -> str:
    return str(v or "").replace("arXiv:", "").replace("arxiv:", "").strip()


def sanitize_model_slug(slug: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_"
                   for c in str(slug)).strip("_")


def safe_int(v: Any) -> Any:
    try:
        if v is None or v == "":
            return None
        n = int(float(v))
        return n
    except Exception:
        return None


def authors_to_string(a: Any) -> str:
    if isinstance(a, list):
        return "; ".join(str(x) for x in a)
    return str(a or "")


# ============================================================
# STEP 1 — focal paper Dimensions metadata
# ============================================================

def load_focal_metadata() -> dict:
    """Build arxiv_id (bare) -> focal paper dimensions dict."""
    df_main = pd.read_csv(PREPRINTS_CSV)
    df_main["paper_arxiv_id"] = df_main["arxiv_id"].astype(str).map(bare_arxiv_id)
    df_main = df_main.drop_duplicates("paper_arxiv_id", keep="first")

    # cross-fill from preprints_with_source_paths (in case some fields differ)
    try:
        df_extra = pd.read_csv(PREPRINTS_SOURCE_CSV)
        df_extra["paper_arxiv_id"] = df_extra["arxiv_id"].astype(str).map(bare_arxiv_id)
        df_extra = df_extra.drop_duplicates("paper_arxiv_id", keep="first")
    except Exception:
        df_extra = pd.DataFrame(columns=["paper_arxiv_id"])

    out = {}
    for _, row in df_main.iterrows():
        aid = row["paper_arxiv_id"]
        out[aid] = {
            "venue_dimensions_id":    row.get("venue_id") or None,
            "preprint_dimensions_id": row.get("preprint_id") or None,
            "conference_name":        row.get("conference_name") or None,
            "first_author_first":     row.get("first_author_first") or None,
            "first_author_last":      row.get("first_author_last") or None,
            "citations_count":        safe_int(row.get("citations_count")),
            "year":                   safe_int(row.get("year")),
        }
    # supplement from source-paths file where missing
    if not df_extra.empty:
        for _, row in df_extra.iterrows():
            aid = row["paper_arxiv_id"]
            d = out.setdefault(aid, {
                "venue_dimensions_id": None, "preprint_dimensions_id": None,
                "conference_name": None, "first_author_first": None,
                "first_author_last": None, "citations_count": None, "year": None,
            })
            if not d.get("venue_dimensions_id"):
                d["venue_dimensions_id"] = row.get("venue_id") or None
            if not d.get("preprint_dimensions_id"):
                d["preprint_dimensions_id"] = row.get("preprint_id") or None
    return out


# ============================================================
# STEP 2 — walk all Phase B outputs, build the master citations DF
# ============================================================

def build_master_dataframe() -> tuple[pd.DataFrame, dict]:
    """Walk every Phase B responses.json and emit one row per citation.

    Returns (master_df, paper_context_lookup), where paper_context_lookup is
    keyed by (source_model, paper_arxiv_id, context_index) and holds the
    per-context fields needed to reconstruct the output JSON later
    (sentences, motivations, etc.).
    """
    judge_path = Path(JUDGE_BASE_DIR)
    source_dirs = sorted(d for d in judge_path.iterdir()
                         if d.is_dir() and d.name.startswith("source__"))
    if not source_dirs:
        raise SystemExit(f"No source__* dirs under {JUDGE_BASE_DIR}")

    print(f"Source-model dirs found: {len(source_dirs)}")

    rows: list[dict] = []
    ctx_lookup: dict[tuple[str, str, int], dict] = {}
    paper_meta_cache: dict[str, dict] = {}  # arxiv_id -> top-level metadata

    t0 = time.time()
    total_files = 0
    for sd in source_dirs:
        files = sorted(glob(os.path.join(str(sd), "*", "responses.json")))
        total_files += len(files)
        print(f"  {sd.name}: {len(files)} papers")

    print(f"Total Phase B files to read: {total_files:,}")
    seen = 0

    for sd in source_dirs:
        files = sorted(glob(os.path.join(str(sd), "*", "responses.json")))
        for fp in files:
            seen += 1
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
            arxiv_id    = bare_arxiv_id(d.get("arxiv_id") or "")
            source_model = d.get("source_model") or ""
            judge_model  = d.get("judge_model") or JUDGE_MODEL_NAME

            paper_meta_cache.setdefault(arxiv_id, {
                "arxiv_id":          d.get("arxiv_id") or arxiv_id,
                "doi":               d.get("doi"),
                "title":             d.get("title"),
                "year":              d.get("year"),
                "venue":             d.get("venue"),
                "preprint_date":     d.get("preprint_date"),
                "first_arxiv_date":  d.get("first_arxiv_date"),
                "latest_arxiv_date": d.get("latest_arxiv_date"),
                "judge_model":       judge_model,
            })

            for r in d.get("results", []) or []:
                ctx_idx = r.get("context_index")
                ctx_key = (source_model, arxiv_id, ctx_idx)
                ctx_lookup[ctx_key] = {
                    "context_index":              ctx_idx,
                    "section":                    r.get("section", "") or "",
                    "source_context_indices":     r.get("source_context_indices", []) or [],
                    "citation_sentence_original": r.get("citation_sentence_original", "") or "",
                    "citation_sentence_filled":   r.get("citation_sentence_filled", "") or "",
                    "before_citation":            r.get("before_citation", "") or "",
                    "after_citation":             r.get("after_citation", "") or "",
                    "masked_paragraph":           r.get("masked_paragraph", "") or "",
                    "num_citations_required":     r.get("num_citations_required"),
                    "num_citations_returned":     r.get("num_citations_returned"),
                    "motivation_judge_original":  r.get("motivation_judge_original"),
                    "motivation_self":            r.get("motivation_self"),
                    "motivation_judge_filled":    r.get("motivation_judge_filled"),
                }

                # ---- ORIGINAL citations from cite_keys + bib_entries ----
                cite_keys = r.get("cite_keys", []) or []
                bib       = r.get("bib_entries", {}) or {}
                for pos, ck in enumerate(cite_keys):
                    be = bib.get(ck) or {}
                    title   = be.get("title", "") or ""
                    authors = authors_to_string(be.get("author", be.get("authors", "")))
                    yraw    = str(be.get("year", be.get("date", "")) or "")
                    year    = yraw[:4] if len(yraw) >= 4 else yraw
                    doi     = be.get("doi", "") or ""
                    venue   = (be.get("journal") or be.get("booktitle")
                               or be.get("venue") or be.get("publisher") or "")
                    rows.append({
                        "source_model":     source_model,
                        "paper_arxiv_id":   arxiv_id,
                        "context_index":    ctx_idx,
                        "position":         pos,
                        "citation_kind":    "original",
                        "cite_key_or_rank": ck,
                        "ref_title":        title,
                        "ref_authors":      authors,
                        "ref_year":         year,
                        "ref_doi":          str(doi),
                        "ref_venue":        str(venue),
                        "match_doi_norm":   normalize_doi(doi),
                        "match_title_norm": normalize_title(title),
                        "year_hint":        safe_int(year),
                    })

                # ---- LLM-GENERATED citations from recommended_papers ----
                recs = r.get("recommended_papers", []) or []
                for pos, rec in enumerate(recs):
                    if not isinstance(rec, dict):
                        continue
                    title   = rec.get("title", "") or ""
                    authors = authors_to_string(rec.get("authors", ""))
                    yraw    = rec.get("year")
                    year    = str(yraw)[:4] if yraw not in (None, "") else ""
                    doi     = str(rec.get("doi", "") or "")
                    venue   = str(rec.get("venue", "") or "")
                    rows.append({
                        "source_model":     source_model,
                        "paper_arxiv_id":   arxiv_id,
                        "context_index":    ctx_idx,
                        "position":         pos,
                        "citation_kind":    "recommended",
                        "cite_key_or_rank": str(pos),
                        "ref_title":        title,
                        "ref_authors":      authors,
                        "ref_year":         year,
                        "ref_doi":          doi,
                        "ref_venue":        venue,
                        "match_doi_norm":   normalize_doi(doi),
                        "match_title_norm": normalize_title(title),
                        "year_hint":        safe_int(year),
                    })

            if seen % PRINT_EVERY == 0:
                rate = seen / max(time.time() - t0, 1e-3)
                print(f"  ... {seen:,}/{total_files:,} files  "
                      f"({rate:.0f}/s, {len(rows):,} rows so far)")

    print(f"Walk done in {time.time() - t0:.1f}s.  Rows: {len(rows):,}")

    df = pd.DataFrame(rows)
    df.insert(0, "row_uid", range(len(df)))
    return df, ctx_lookup, paper_meta_cache


# ============================================================
# STEP 3 — BigQuery: upload, JOIN, download
# ============================================================

def run_bigquery_match(master_df: pd.DataFrame) -> pd.DataFrame:
    client = bigquery.Client(project=BQ_PROJECT, location=BQ_LOCATION)

    # If SKIP_BQ_JOIN=1 the upload + match SQL are skipped and we just
    # re-download the already-populated citations_matched table. Useful when
    # only the JSON-emit step changed.
    if os.getenv("SKIP_BQ_JOIN", "0") == "1":
        print(f"SKIP_BQ_JOIN=1: skipping upload+match, "
              f"downloading existing `{BQ_OUTPUT_TABLE}`")
        t0 = time.time()
        matched = client.query(
            f"SELECT * FROM `{BQ_OUTPUT_TABLE}`"
        ).result().to_dataframe(create_bqstorage_client=False)
        print(f"  download done in {time.time() - t0:.1f}s, {len(matched):,} rows")
        return matched

    # Ensure dataset exists.
    ds_ref = bigquery.Dataset(f"{BQ_PROJECT}.{BQ_DATASET}")
    ds_ref.location = BQ_LOCATION
    try:
        client.get_dataset(ds_ref)
    except Exception:
        client.create_dataset(ds_ref, exists_ok=True)
        print(f"Created dataset {BQ_PROJECT}.{BQ_DATASET}")

    # Upload the smaller "input" projection we need on BQ side.
    upload = master_df[[
        "row_uid", "citation_kind", "match_doi_norm", "match_title_norm", "year_hint"
    ]].copy()
    upload["year_hint"] = pd.to_numeric(upload["year_hint"], errors="coerce").astype("Int64")
    upload["match_doi_norm"]   = upload["match_doi_norm"].fillna("").astype(str)
    upload["match_title_norm"] = upload["match_title_norm"].fillna("").astype(str)
    upload["citation_kind"]    = upload["citation_kind"].fillna("").astype(str)

    schema = [
        bigquery.SchemaField("row_uid",          "INT64"),
        bigquery.SchemaField("citation_kind",    "STRING"),
        bigquery.SchemaField("match_doi_norm",   "STRING"),
        bigquery.SchemaField("match_title_norm", "STRING"),
        bigquery.SchemaField("year_hint",        "INT64"),
    ]
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=schema,
    )
    print(f"Uploading {len(upload):,} rows -> {BQ_INPUT_TABLE}")
    t0 = time.time()
    client.load_table_from_dataframe(upload, BQ_INPUT_TABLE, job_config=job_config).result()
    print(f"  upload done in {time.time() - t0:.1f}s")

    # Symmetric match SQL: DOI first, then title fallback for BOTH
    # originals and recommended rows. This is the "dointitle" variant.
    #
    # SQL dim_title_norm pipeline mirrors Python normalize_title:
    #   - NORMALIZE(NFKD)             decompose accented chars
    #   - strip \p{Mn}+               drop combining marks (folds é -> e)
    #   - LOWER
    #   - strip LaTeX commands like  \textbf{X} -> X  and  \alpha -> ''
    #   - replace non-[a-z0-9 ] -> space
    #   - drop {the,a,an}
    #   - collapse whitespace
    sql = f"""
CREATE OR REPLACE TABLE `{BQ_OUTPUT_TABLE}` AS
WITH
  to_match AS (
    SELECT row_uid, citation_kind, match_doi_norm, match_title_norm, year_hint
    FROM `{BQ_INPUT_TABLE}`
  ),
  pubs AS (
    SELECT
      p.id                                                                    AS dim_id,
      p.doi                                                                   AS dim_doi,
      LOWER(p.doi)                                                            AS dim_doi_norm,
      p.title.preferred                                                       AS dim_title,
      TRIM(REGEXP_REPLACE(
        REGEXP_REPLACE(
          REGEXP_REPLACE(
            REGEXP_REPLACE(
              REGEXP_REPLACE(
                LOWER(
                  REGEXP_REPLACE(
                    NORMALIZE(p.title.preferred, NFKD),
                    r'\\pM+', ''
                  )
                ),
                r'\\\\[a-z]+\\{{([^}}]*)\\}}', r'\\1'                            -- \\textbf{{X}} -> X
              ),
              r'\\\\[a-z]+|[{{}}]', ''                                          -- \\alpha -> ''  (also strip stray braces)
            ),
            r'[^a-z0-9\\s]', ' '
          ),
          r'\\b(the|a|an)\\b', ' '
        ),
        r'\\s+', ' '
      ))                                                                      AS dim_title_norm,
      p.year                                                                  AS dim_year,
      COALESCE(p.conference.name, p.source.title)                             AS dim_venue,
      p.citations_count                                                       AS dim_citations,
      ARRAY_LENGTH(p.authors)                                                 AS dim_teamsize,
      LOWER(p.type)                                                           AS dim_type
    FROM `{DIMENSIONS_TABLE}` p
    WHERE p.doi IS NOT NULL OR p.title.preferred IS NOT NULL
  ),
  doi_hits AS (
    SELECT
      t.row_uid, 'doi' AS match_method,
      p.dim_id, p.dim_doi, p.dim_doi_norm, p.dim_title, p.dim_year, p.dim_venue,
      p.dim_citations, p.dim_teamsize, p.dim_type,
      ROW_NUMBER() OVER (
        PARTITION BY t.row_uid
        ORDER BY
          CASE p.dim_type WHEN 'article'    THEN 0
                          WHEN 'proceeding' THEN 1
                          WHEN 'chapter'    THEN 2
                          WHEN 'monograph'  THEN 3
                          WHEN 'book'       THEN 4
                          WHEN 'preprint'   THEN 5
                          ELSE 9 END,
          ABS(IFNULL(p.dim_year, 9999) - IFNULL(t.year_hint, 9999)),
          IFNULL(p.dim_citations, 0) DESC
      ) AS rn
    FROM to_match t
    JOIN pubs p
      ON t.match_doi_norm != '' AND p.dim_doi_norm = t.match_doi_norm
  ),
  doi_winners AS (
    SELECT row_uid, match_method, dim_id, dim_doi, dim_doi_norm, dim_title,
           dim_year, dim_venue, dim_citations, dim_teamsize, dim_type
    FROM doi_hits WHERE rn = 1
  ),
  title_hits AS (
    SELECT
      t.row_uid, 'title' AS match_method,
      p.dim_id, p.dim_doi, p.dim_doi_norm, p.dim_title, p.dim_year, p.dim_venue,
      p.dim_citations, p.dim_teamsize, p.dim_type,
      ROW_NUMBER() OVER (
        PARTITION BY t.row_uid
        ORDER BY
          CASE p.dim_type WHEN 'article'    THEN 0
                          WHEN 'proceeding' THEN 1
                          WHEN 'chapter'    THEN 2
                          WHEN 'monograph'  THEN 3
                          WHEN 'book'       THEN 4
                          WHEN 'preprint'   THEN 5
                          ELSE 9 END,
          ABS(IFNULL(p.dim_year, 9999) - IFNULL(t.year_hint, 9999)),
          IFNULL(p.dim_citations, 0) DESC
      ) AS rn
    FROM to_match t
    JOIN pubs p
      ON t.match_title_norm != '' AND p.dim_title_norm = t.match_title_norm
    WHERE t.row_uid NOT IN (SELECT row_uid FROM doi_winners)
  ),
  title_winners AS (
    SELECT row_uid, match_method, dim_id, dim_doi, dim_doi_norm, dim_title,
           dim_year, dim_venue, dim_citations, dim_teamsize, dim_type
    FROM title_hits WHERE rn = 1
  )
SELECT * FROM doi_winners
UNION ALL
SELECT * FROM title_winners
"""
    print("Running match SQL on BigQuery...")
    t0 = time.time()
    job = client.query(sql)
    job.result()
    bytes_billed = job.total_bytes_billed or 0
    print(f"  match SQL done in {time.time() - t0:.1f}s, "
          f"bytes billed: {bytes_billed / 1e9:.2f} GB "
          f"(~${bytes_billed / 1e12 * 6.25:.3f})")

    print(f"Downloading matched table -> pandas")
    t0 = time.time()
    matched = client.query(f"SELECT * FROM `{BQ_OUTPUT_TABLE}`").result().to_dataframe(
        create_bqstorage_client=False)
    print(f"  download done in {time.time() - t0:.1f}s, {len(matched):,} rows")
    return matched


def run_bigquery_match_focals(focal_meta: dict, paper_arxiv_ids: list[str]) -> dict:
    """Resolve focal-paper Dimensions metadata (citations_count, teamsize, etc.)
    by direct ID lookup against publications, for both the venue (published)
    and preprint variants. Returns a dict keyed by bare arxiv_id."""
    client = bigquery.Client(project=BQ_PROJECT, location=BQ_LOCATION)

    if os.getenv("SKIP_BQ_JOIN", "0") == "1":
        print(f"SKIP_BQ_JOIN=1: skipping focal upload+match, "
              f"downloading existing `{BQ_FOCAL_OUTPUT_TABLE}`")
        try:
            t0 = time.time()
            df = client.query(
                f"SELECT * FROM `{BQ_FOCAL_OUTPUT_TABLE}`"
            ).result().to_dataframe(create_bqstorage_client=False)
            print(f"  download done in {time.time() - t0:.1f}s, {len(df):,} rows")
        except Exception as e:
            print(f"  WARN: could not read existing table ({e}); returning empty")
            return {}
    else:
        # Stage: one row per focal paper, with its two possible Dimensions IDs.
        rows = []
        for aid in paper_arxiv_ids:
            m = focal_meta.get(aid) or {}
            rows.append({
                "paper_arxiv_id":         aid,
                "venue_dimensions_id":    m.get("venue_dimensions_id") or None,
                "preprint_dimensions_id": m.get("preprint_dimensions_id") or None,
            })
        focal_df = pd.DataFrame(rows)
        schema = [
            bigquery.SchemaField("paper_arxiv_id",         "STRING"),
            bigquery.SchemaField("venue_dimensions_id",    "STRING"),
            bigquery.SchemaField("preprint_dimensions_id", "STRING"),
        ]
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE", schema=schema,
        )
        print(f"Uploading {len(focal_df):,} focal-paper rows -> {BQ_FOCAL_INPUT_TABLE}")
        t0 = time.time()
        client.load_table_from_dataframe(
            focal_df, BQ_FOCAL_INPUT_TABLE, job_config=job_config
        ).result()
        print(f"  upload done in {time.time() - t0:.1f}s")

        sql = f"""
CREATE OR REPLACE TABLE `{BQ_FOCAL_OUTPUT_TABLE}` AS
WITH
  focals AS (SELECT * FROM `{BQ_FOCAL_INPUT_TABLE}`),
  pubs AS (
    SELECT
      p.id                                        AS dim_id,
      p.doi                                       AS dim_doi,
      p.title.preferred                           AS dim_title,
      p.year                                      AS dim_year,
      COALESCE(p.conference.name, p.source.title) AS dim_venue,
      p.citations_count                           AS dim_citations,
      ARRAY_LENGTH(p.authors)                     AS dim_teamsize,
      LOWER(p.type)                               AS dim_type
    FROM `{DIMENSIONS_TABLE}` p
    WHERE p.id IN (
      SELECT venue_dimensions_id    FROM focals WHERE venue_dimensions_id    IS NOT NULL
      UNION DISTINCT
      SELECT preprint_dimensions_id FROM focals WHERE preprint_dimensions_id IS NOT NULL
    )
  )
SELECT
  f.paper_arxiv_id,

  f.venue_dimensions_id,
  v.dim_doi       AS venue_dim_doi,
  v.dim_title     AS venue_dim_title,
  v.dim_year      AS venue_dim_year,
  v.dim_venue     AS venue_dim_venue,
  v.dim_citations AS venue_dim_citations,
  v.dim_teamsize  AS venue_dim_teamsize,
  v.dim_type      AS venue_dim_type,

  f.preprint_dimensions_id,
  pp.dim_doi       AS preprint_dim_doi,
  pp.dim_title     AS preprint_dim_title,
  pp.dim_year      AS preprint_dim_year,
  pp.dim_venue     AS preprint_dim_venue,
  pp.dim_citations AS preprint_dim_citations,
  pp.dim_teamsize  AS preprint_dim_teamsize,
  pp.dim_type      AS preprint_dim_type
FROM focals f
LEFT JOIN pubs v  ON v.dim_id  = f.venue_dimensions_id
LEFT JOIN pubs pp ON pp.dim_id = f.preprint_dimensions_id
"""
        print("Running focal-paper match SQL on BigQuery...")
        t0 = time.time()
        job = client.query(sql)
        job.result()
        bytes_billed = job.total_bytes_billed or 0
        print(f"  focal match SQL done in {time.time() - t0:.1f}s, "
              f"bytes billed: {bytes_billed / 1e9:.2f} GB "
              f"(~${bytes_billed / 1e12 * 6.25:.3f})")

        t0 = time.time()
        df = client.query(
            f"SELECT * FROM `{BQ_FOCAL_OUTPUT_TABLE}`"
        ).result().to_dataframe(create_bqstorage_client=False)
        print(f"  focal download done in {time.time() - t0:.1f}s, {len(df):,} rows")

    out: dict[str, dict] = {}
    for r in df.to_dict("records"):
        out[r["paper_arxiv_id"]] = r
    return out


# ============================================================
# STEP 4 — merge, summarize, emit per-source-model JSON
# ============================================================

def _nan_to_none(v: Any) -> Any:
    try:
        if v is None:
            return None
        if isinstance(v, float) and (v != v):  # NaN check
            return None
        return v
    except Exception:
        return v


def _agreement_flags(row: dict) -> tuple:
    """Compute (doi_provided, doi_agrees, year_provided, year_agrees) for one
    merged row. *_agrees is None when the row was unmatched OR the relevant
    field wasn't supplied by the LLM/bib OR isn't present on the Dimensions
    side. _provided reflects only whether the LLM/bib side carried the value."""
    matched = _nan_to_none(row.get("dim_id")) is not None
    claimed_doi = str(row.get("match_doi_norm") or "").strip()
    dim_doi     = str(row.get("dim_doi_norm")   or "").strip()
    doi_provided = bool(claimed_doi)

    year_hint = row.get("year_hint")
    if year_hint is None or (isinstance(year_hint, float) and year_hint != year_hint):
        year_provided = False
        rh = None
    else:
        year_provided = True
        rh = safe_int(year_hint)
    dh = safe_int(row.get("dim_year"))

    if not matched:
        return doi_provided, None, year_provided, None

    if doi_provided and dim_doi:
        doi_agrees = (claimed_doi == dim_doi)
    else:
        doi_agrees = None

    year_agrees = (rh == dh) if (rh is not None and dh is not None) else None
    return doi_provided, doi_agrees, year_provided, year_agrees


def _row_to_orig_item(row: dict) -> dict:
    doi_provided, doi_agrees, year_provided, year_agrees = _agreement_flags(row)
    return {
        "position":                    int(row["position"]),
        "cite_key":                    row["cite_key_or_rank"],
        "ref_title":                   row["ref_title"],
        "ref_authors":                 row["ref_authors"],
        "ref_year":                    row["ref_year"],
        "ref_doi":                     row["ref_doi"],
        "ref_venue":                   row["ref_venue"],
        "match_method":                _nan_to_none(row.get("match_method")),
        "dimensions_id":               _nan_to_none(row.get("dim_id")),
        "dimensions_doi":              _nan_to_none(row.get("dim_doi")),
        "dimensions_title":            _nan_to_none(row.get("dim_title")),
        "dimensions_year":             safe_int(row.get("dim_year")),
        "dimensions_venue":            _nan_to_none(row.get("dim_venue")),
        "dimensions_citations_count":  safe_int(row.get("dim_citations")),
        "dimensions_teamsize":         safe_int(row.get("dim_teamsize")),
        "dimensions_type":             _nan_to_none(row.get("dim_type")),
        "doi_provided":                doi_provided,
        "doi_agrees_with_match":       doi_agrees,
        "year_provided":               year_provided,
        "year_agrees_with_match":      year_agrees,
    }


def _row_to_rec_item(row: dict) -> dict:
    doi_provided, doi_agrees, year_provided, year_agrees = _agreement_flags(row)
    return {
        "position":                    int(row["position"]),
        "rec_title":                   row["ref_title"],
        "rec_authors":                 row["ref_authors"],
        "rec_year":                    safe_int(row["ref_year"]) if row["ref_year"] else None,
        "rec_doi":                     row["ref_doi"],
        "rec_venue":                   row["ref_venue"],
        "match_method":                _nan_to_none(row.get("match_method")),
        "dimensions_id":               _nan_to_none(row.get("dim_id")),
        "dimensions_doi":              _nan_to_none(row.get("dim_doi")),
        "dimensions_title":            _nan_to_none(row.get("dim_title")),
        "dimensions_year":             safe_int(row.get("dim_year")),
        "dimensions_venue":            _nan_to_none(row.get("dim_venue")),
        "dimensions_citations_count":  safe_int(row.get("dim_citations")),
        "dimensions_teamsize":         safe_int(row.get("dim_teamsize")),
        "dimensions_type":             _nan_to_none(row.get("dim_type")),
        "doi_provided":                doi_provided,
        "doi_agrees_with_match":       doi_agrees,
        "year_provided":               year_provided,
        "year_agrees_with_match":      year_agrees,
    }


def emit_outputs(merged: pd.DataFrame,
                 ctx_lookup: dict,
                 paper_meta_cache: dict,
                 focal_meta: dict,
                 focal_bq: dict) -> None:

    os.makedirs(OUT_DIR, exist_ok=True)

    # Compute agreement columns once at the DataFrame level so we can
    # aggregate per source model / kind without re-walking row by row.
    claimed_doi = merged["match_doi_norm"].fillna("").astype(str).str.strip()
    dim_doi     = merged["dim_doi_norm"].fillna("").astype(str).str.strip()
    merged["doi_provided"] = claimed_doi.ne("")
    merged["doi_agrees_with_match"] = (
        merged["dim_id"].notna() & merged["doi_provided"] & dim_doi.ne("")
        & (claimed_doi == dim_doi)
    )
    # year_provided / year_agrees
    yh = pd.to_numeric(merged["year_hint"], errors="coerce")
    dy = pd.to_numeric(merged["dim_year"],  errors="coerce")
    merged["year_provided"] = yh.notna()
    merged["year_agrees_with_match"] = (
        merged["dim_id"].notna() & yh.notna() & dy.notna() & (yh == dy)
    )

    # ===== Overall match summary =====
    total = len(merged)
    by_method_overall = merged["match_method"].fillna("unmatched").value_counts()
    print("\n" + "=" * 60)
    print("OVERALL match summary (originals + recommended, all source models)")
    print("=" * 60)
    for m, c in by_method_overall.items():
        print(f"  {m:12s} : {c:>10,}  ({c/total*100:6.2f}%)")
    print(f"  {'TOTAL':12s} : {total:>10,}")

    print("\nBy citation_kind:")
    for kind in ["original", "recommended"]:
        sub = merged[merged["citation_kind"] == kind]
        if len(sub) == 0:
            continue
        s = sub["match_method"].fillna("unmatched").value_counts()
        matched_n = int(sub["dim_id"].notna().sum())
        print(f"  {kind:12s} n={len(sub):,}  matched={matched_n:,} "
              f"({matched_n/len(sub)*100:.2f}%)")
        for m, c in s.items():
            print(f"      {m:10s} : {c:>10,}  ({c/len(sub)*100:6.2f}%)")

    print("\nBy source model & kind:")
    for sm in sorted(merged["source_model"].unique()):
        for kind in ["original", "recommended"]:
            sub = merged[(merged["source_model"] == sm) &
                         (merged["citation_kind"] == kind)]
            if len(sub) == 0:
                continue
            n_match = int(sub["dim_id"].notna().sum())
            print(f"  {sm:38s} / {kind:11s}  "
                  f"{n_match:>7,}/{len(sub):<7,}  "
                  f"({n_match/len(sub)*100:6.2f}%)")

    # ===== Match-quality metrics: DOI / year agreement on matched rows =====
    print("\n" + "=" * 60)
    print("QUALITY METRICS on MATCHED rows (DOI / year agreement)")
    print("=" * 60)
    print("Conventions:")
    print("  DOI provided     : LLM/bib supplied a non-empty DOI")
    print("  DOI agrees       : provided DOI == matched Dimensions DOI (norm)")
    print("  Year provided    : LLM/bib supplied a year")
    print("  Year agrees      : provided year == matched Dimensions year")
    print("Percentages of 'agrees' are out of matched rows where the field was provided.")

    def _qual_block(sub: pd.DataFrame, label: str):
        n_m = int(sub["dim_id"].notna().sum())
        if n_m == 0:
            print(f"  {label}: (no matches)")
            return
        sub_m = sub[sub["dim_id"].notna()]
        n_doi_p   = int(sub_m["doi_provided"].sum())
        n_doi_a   = int(sub_m["doi_agrees_with_match"].sum())
        n_year_p  = int(sub_m["year_provided"].sum())
        n_year_a  = int(sub_m["year_agrees_with_match"].sum())
        pct_doi_p = n_doi_p / n_m * 100
        pct_doi_a = (n_doi_a / n_doi_p * 100) if n_doi_p else 0.0
        pct_y_p   = n_year_p / n_m * 100
        pct_y_a   = (n_year_a / n_year_p * 100) if n_year_p else 0.0
        print(f"  {label}  matched={n_m:,}")
        print(f"     DOI provided        : {n_doi_p:>8,}/{n_m:<8,} ({pct_doi_p:6.2f}%)")
        print(f"     DOI agrees w/ match : {n_doi_a:>8,}/{n_doi_p:<8,} ({pct_doi_a:6.2f}% of provided)")
        print(f"     Year provided       : {n_year_p:>8,}/{n_m:<8,} ({pct_y_p:6.2f}%)")
        print(f"     Year agrees w/ match: {n_year_a:>8,}/{n_year_p:<8,} ({pct_y_a:6.2f}% of provided)")

    print("\nBy citation_kind:")
    for kind in ["original", "recommended"]:
        sub = merged[merged["citation_kind"] == kind]
        if len(sub):
            _qual_block(sub, f"{kind:11s}")

    print("\nRecommended (LLM-generated) by source model:")
    for sm in sorted(merged["source_model"].unique()):
        sub = merged[(merged["source_model"] == sm) &
                     (merged["citation_kind"] == "recommended")]
        if len(sub):
            _qual_block(sub, f"{sm:38s}")

    # ===== Emit per-source-model JSON =====
    print("\nWriting per-source-model JSON files...")

    for sm, sm_df in merged.groupby("source_model"):
        src_slug = sanitize_model_slug(sm)
        out_name = f"source__{src_slug}__judge__{JUDGE_SLUG}.citations_with_dimensions.json"
        out_path = os.path.join(OUT_DIR, out_name)

        # match summary for this file
        sm_total = len(sm_df)
        sm_method_counts = sm_df["match_method"].fillna("unmatched").value_counts().to_dict()
        match_summary = {
            m: {"count": int(c), "pct": float(c / sm_total * 100)}
            for m, c in sm_method_counts.items()
        }

        papers_out = []
        sm_df_sorted = sm_df.sort_values(["paper_arxiv_id", "context_index",
                                          "citation_kind", "position"])

        for paper_aid, paper_df in sm_df_sorted.groupby("paper_arxiv_id", sort=True):
            meta = paper_meta_cache.get(paper_aid, {})
            focal = focal_meta.get(paper_aid, {})

            results_out = []
            for ctx_idx, ctx_df in paper_df.groupby("context_index", sort=True):
                ctx_info = ctx_lookup.get((sm, paper_aid, ctx_idx), {})

                orig_df = ctx_df[ctx_df["citation_kind"] == "original"]
                rec_df  = ctx_df[ctx_df["citation_kind"] == "recommended"]

                orig_items = [_row_to_orig_item(r) for r in
                              orig_df.sort_values("position").to_dict("records")]
                rec_items  = [_row_to_rec_item(r) for r in
                              rec_df.sort_values("position").to_dict("records")]

                results_out.append({
                    "context_index":              _nan_to_none(ctx_info.get("context_index", ctx_idx)),
                    "section":                    ctx_info.get("section", ""),
                    "source_context_indices":     ctx_info.get("source_context_indices", []),
                    "citation_sentence_original": ctx_info.get("citation_sentence_original", ""),
                    "citation_sentence_filled":   ctx_info.get("citation_sentence_filled", ""),
                    "before_citation":            ctx_info.get("before_citation", ""),
                    "after_citation":             ctx_info.get("after_citation", ""),
                    "masked_paragraph":           ctx_info.get("masked_paragraph", ""),
                    "num_citations_required":     _nan_to_none(ctx_info.get("num_citations_required")),
                    "num_citations_returned":     _nan_to_none(ctx_info.get("num_citations_returned")),
                    "motivation_judge_original":  ctx_info.get("motivation_judge_original"),
                    "motivation_self":            ctx_info.get("motivation_self"),
                    "motivation_judge_filled":    ctx_info.get("motivation_judge_filled"),
                    "original_citations": {
                        "count":         len(orig_items),
                        "matched_count": int(orig_df["dim_id"].notna().sum()),
                        "items":         orig_items,
                    },
                    "llm_generated_citations": {
                        "count":         len(rec_items),
                        "matched_count": int(rec_df["dim_id"].notna().sum()),
                        "items":         rec_items,
                    },
                })

            bq_focal = focal_bq.get(paper_aid) or {}

            # For focal papers, if the BQ join did find the record (i.e., the
            # paper exists in publications) but citations_count is null — which
            # is common for very fresh 2025 papers Dimensions hasn't backfilled
            # yet — default to 0 instead of null. We use dim_title as the
            # "row was found" indicator (more reliable than dim_doi, which is
            # sometimes absent for legit records).
            def _matched_citations(prefix: str):
                if _nan_to_none(bq_focal.get(f"{prefix}_title")) is None:
                    return None  # paper not found in publications -> truly unknown
                n = safe_int(bq_focal.get(f"{prefix}_citations"))
                return 0 if n is None else n

            focal_out = {
                "venue_dimensions_id":    _nan_to_none((focal or {}).get("venue_dimensions_id")),
                "preprint_dimensions_id": _nan_to_none((focal or {}).get("preprint_dimensions_id")),
                "conference_name":        _nan_to_none((focal or {}).get("conference_name")),
                "first_author_first":     _nan_to_none((focal or {}).get("first_author_first")),
                "first_author_last":      _nan_to_none((focal or {}).get("first_author_last")),
                "year":                   safe_int((focal or {}).get("year")),

                # venue (published version) Dimensions fields
                "venue_dimensions_doi":             _nan_to_none(bq_focal.get("venue_dim_doi")),
                "venue_dimensions_title":           _nan_to_none(bq_focal.get("venue_dim_title")),
                "venue_dimensions_year":            safe_int(bq_focal.get("venue_dim_year")),
                "venue_dimensions_venue":           _nan_to_none(bq_focal.get("venue_dim_venue")),
                "venue_dimensions_citations_count": _matched_citations("venue_dim"),
                "venue_dimensions_teamsize":        safe_int(bq_focal.get("venue_dim_teamsize")),
                "venue_dimensions_type":            _nan_to_none(bq_focal.get("venue_dim_type")),

                # preprint (arXiv version) Dimensions fields
                "preprint_dimensions_doi":             _nan_to_none(bq_focal.get("preprint_dim_doi")),
                "preprint_dimensions_title":           _nan_to_none(bq_focal.get("preprint_dim_title")),
                "preprint_dimensions_year":            safe_int(bq_focal.get("preprint_dim_year")),
                "preprint_dimensions_venue":           _nan_to_none(bq_focal.get("preprint_dim_venue")),
                "preprint_dimensions_citations_count": _matched_citations("preprint_dim"),
                "preprint_dimensions_teamsize":        safe_int(bq_focal.get("preprint_dim_teamsize")),
                "preprint_dimensions_type":            _nan_to_none(bq_focal.get("preprint_dim_type")),
            }

            papers_out.append({
                "arxiv_id":          meta.get("arxiv_id", paper_aid),
                "doi":               meta.get("doi"),
                "title":             meta.get("title"),
                "year":              meta.get("year"),
                "venue":             meta.get("venue"),
                "preprint_date":     meta.get("preprint_date"),
                "first_arxiv_date":  meta.get("first_arxiv_date"),
                "latest_arxiv_date": meta.get("latest_arxiv_date"),
                "focal_paper_dimensions": focal_out,
                "num_contexts": len(results_out),
                "results":      results_out,
            })

        out_blob = {
            "judge_model":           JUDGE_MODEL_NAME,
            "source_model":          sm,
            "judge_base_dir":        JUDGE_BASE_DIR,
            "bq_input_table":        BQ_INPUT_TABLE,
            "bq_output_table":       BQ_OUTPUT_TABLE,
            "dimensions_table":      DIMENSIONS_TABLE,
            "num_papers":            len(papers_out),
            "num_citation_rows":     int(sm_total),
            "match_method_summary":  match_summary,
            "papers":                papers_out,
        }

        payload = json.dumps(
            out_blob,
            ensure_ascii=False, indent=2,
            default=lambda o: None if (isinstance(o, float) and o != o) else str(o),
        )
        # Strip lone surrogates that occasionally leak in from LaTeX bib_entries.
        with open(out_path, "wb") as f:
            f.write(payload.encode("utf-8", errors="replace"))
        print(f"  wrote {out_path}  "
              f"({len(papers_out):,} papers, {sm_total:,} citation rows)")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("Step 03 — Dimensions match for Step-02 Phase-B outputs")
    print(f"Judge base dir: {JUDGE_BASE_DIR}")
    print(f"Output dir:     {OUT_DIR}")
    print(f"BQ project:     {BQ_PROJECT}")
    print(f"BQ dataset:     {BQ_DATASET}")
    print("=" * 72)

    focal_meta = load_focal_metadata()
    print(f"Loaded focal-paper metadata for {len(focal_meta):,} arxiv ids")

    master_df, ctx_lookup, paper_meta_cache = build_master_dataframe()
    print(f"Master citations DF: {len(master_df):,} rows  "
          f"(originals={int((master_df['citation_kind']=='original').sum()):,}, "
          f"recommended={int((master_df['citation_kind']=='recommended').sum()):,})")

    matched_df = run_bigquery_match(master_df)

    merged = master_df.merge(matched_df, on="row_uid", how="left")
    print(f"Merged matched table back: {len(merged):,} rows, "
          f"{int(merged['dim_id'].notna().sum()):,} have a Dimensions match.")

    paper_arxiv_ids = sorted(set(paper_meta_cache.keys()))
    focal_bq = run_bigquery_match_focals(focal_meta, paper_arxiv_ids)
    print(f"Focal-paper BQ map: {len(focal_bq):,} papers")

    emit_outputs(merged, ctx_lookup, paper_meta_cache, focal_meta, focal_bq)
    print("\nDone.")


if __name__ == "__main__":
    main()
