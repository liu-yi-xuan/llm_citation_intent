"""
Step 04 — Build author-dyad parquet tables and researcher-metadata parquet from
the Step-03 DOI+title-matched Gemini-judged outputs.

Inputs:
  data/dimensions_dointitle_matched_judge_gemini/*.json

BigQuery side (cached in <project>.<dataset> so subsequent runs
can `SKIP_BQ_PUBS=1` / `SKIP_BQ_RESEARCHERS=1` to skip rescans):

  pubs_to_lookup            staging (dim_ids we need authors for)
  pubs_authors              per-publication: first/last author (researcher_id,
                            first/last name, orcid), arxiv_id, doi, year, venue,
                            type, teamsize, citations_count
  researchers_to_lookup     staging (non-null researcher_ids from pubs_authors)
  researchers_metadata      per researcher_id:
                              first_name, last_name, orcid_ids, current_research_org
                              first_publication_year, last_publication_year
                              total_publications_until_2024
                              total_citations_received_before_2025
                              career_age_at_cutoff   ( = 2025 - first_pub_year )
                              first_affiliation_country

Outputs:
  data/author_dyads_judge_gemini/pub_authors.parquet
  data/author_dyads_judge_gemini/researcher_metadata.parquet
  data/author_dyads_judge_gemini/original_dyads.parquet      (shared across all 6 source models)
  data/author_dyads_judge_gemini/llm_generated__{src_slug}.parquet  (6 files)

One dyad row carries focal-paper authors (first + last) AND cited-paper authors
(first + last), so downstream analysis can explode into the 4 author-author
pairs as needed. Author longitudinal metadata (career age, productivity,
citations, first-ever affiliation country) lives only in
researcher_metadata.parquet -- join by researcher_id when needed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery


# ============================================================
# CONFIG
# ============================================================

JUDGE_OUT_DIR             = "data/dimensions_dointitle_matched_judge_gemini"
JUDGE_NAME                = "google/gemini-3-flash-preview"
JUDGE_SLUG                = "google_gemini-3-flash-preview"
OUT_DIR                   = "data/author_dyads_judge_gemini"

# Configure via env (or config.yml — see config.example.yml). Anonymized defaults.
BQ_PROJECT                = os.environ.get("BQ_PROJECT", "your-gcp-project")
BQ_DATASET                = os.environ.get("BQ_DATASET", "your_dataset")
BQ_PUB_LOOKUP_TABLE       = f"{BQ_PROJECT}.{BQ_DATASET}.pubs_to_lookup"
BQ_PUB_AUTHORS_TABLE      = f"{BQ_PROJECT}.{BQ_DATASET}.pubs_authors"
BQ_RESEARCHERS_LOOKUP_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.researchers_to_lookup"
BQ_RESEARCHERS_OUT_TABLE  = f"{BQ_PROJECT}.{BQ_DATASET}.researchers_metadata"
BQ_LOCATION               = "US"

DIMENSIONS_PUBS_TABLE        = "dimensions-ai.data_analytics.publications"
DIMENSIONS_RESEARCHERS_TABLE = "dimensions-ai.data_analytics.researchers"

YEAR_CUTOFF             = 2024   # publications up to and including this year
CITATION_YEAR_CUTOFF    = 2025   # citations strictly before this year
CAREER_REF_YEAR         = 2025

USD_PER_TB              = 6.25   # rough BigQuery on-demand pricing


def sanitize_model_slug(slug: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(slug)).strip("_")


def bare_arxiv_id(v: Any) -> str:
    return str(v or "").replace("arXiv:", "").replace("arxiv:", "").strip()


def safe_int(v: Any):
    """Coerce a value to Python int or None. Handles NaN, empty strings,
    pandas/numpy nullable ints, and non-numeric strings."""
    if v is None:
        return None
    if isinstance(v, float):
        if v != v:   # NaN
            return None
        return int(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except (TypeError, ValueError):
            return None
    # numpy / pandas types
    try:
        if pd.isna(v):
            return None
        return int(v)
    except Exception:
        try:
            return int(v)
        except Exception:
            return None


def safe_str(v: Any):
    """Coerce to str or None, treating NaN/empty/missing as None."""
    if v is None:
        return None
    if isinstance(v, float) and v != v:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v)
    return s if s != "" else None


# ============================================================
# STEP 1 — WALK THE 6 JSON FILES
# ============================================================

def walk_jsons():
    """Single pass over the 6 dointitle JSON files. Returns five things:

    pub_ids:           set of dim_ids we need author info for
    paper_meta:        dict by focal_arxiv_id -> paper metadata
    ctx_meta:          dict by (focal_arxiv_id, ctx_idx) -> shared context fields
                       (section, motivation_judge_original, etc.)
    original_records:  list of dicts, one per (focal_arxiv, ctx_idx, position),
                       deduped across source models (since originals are identical)
    llm_records:       dict by source_model -> list of dicts (one per LLM recommendation)
    """
    files = sorted(glob(os.path.join(
        JUDGE_OUT_DIR, "source__*__judge__*.citations_with_dimensions.json")))
    if not files:
        raise SystemExit(f"No source__*.json found under {JUDGE_OUT_DIR}")

    print(f"Source-model JSON files: {len(files)}")

    pub_ids: set[str] = set()
    paper_meta: dict[str, dict] = {}
    ctx_meta:   dict[tuple, dict] = {}
    original_records: list[dict] = []
    original_seen: set[tuple] = set()
    llm_records: dict[str, list[dict]] = {}

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        sm = d.get("source_model") or ""
        llm_records.setdefault(sm, [])
        n_papers = len(d.get("papers", []))
        print(f"  {os.path.basename(fp)}: {n_papers} papers")

        for paper in d.get("papers", []):
            arxiv = bare_arxiv_id(paper.get("arxiv_id"))
            focal_dim_block = paper.get("focal_paper_dimensions") or {}
            focal_dim_id   = focal_dim_block.get("venue_dimensions_id")
            focal_prep_id  = focal_dim_block.get("preprint_dimensions_id")
            if focal_dim_id:
                pub_ids.add(focal_dim_id)
            if focal_prep_id:
                pub_ids.add(focal_prep_id)

            paper_meta.setdefault(arxiv, {
                "focal_arxiv_id":            arxiv,
                "focal_dim_id":              focal_dim_id,
                "focal_preprint_dim_id":     focal_prep_id,
                "focal_doi":                 paper.get("doi"),
                "focal_title":               paper.get("title"),
                "focal_year":                paper.get("year"),
                "focal_venue":               paper.get("venue"),
                "focal_preprint_date":       paper.get("preprint_date"),
                "focal_first_arxiv_date":    paper.get("first_arxiv_date"),
                "focal_latest_arxiv_date":   paper.get("latest_arxiv_date"),
                "focal_csv_first_author_first": focal_dim_block.get("first_author_first"),
                "focal_csv_first_author_last":  focal_dim_block.get("first_author_last"),
            })

            for ctx in paper.get("results", []) or []:
                ctx_idx = ctx.get("context_index")
                ctx_key = (arxiv, ctx_idx)
                if ctx_key not in ctx_meta:
                    mjo = ctx.get("motivation_judge_original") or {}
                    ctx_meta[ctx_key] = {
                        "context_index":              ctx_idx,
                        "section":                    ctx.get("section"),
                        "num_citations_required":     ctx.get("num_citations_required"),
                        "citation_sentence_original": ctx.get("citation_sentence_original"),
                        "motivation_judge_original":  mjo.get("motivation") if isinstance(mjo, dict) else None,
                        "confidence_judge_original":  mjo.get("confidence") if isinstance(mjo, dict) else None,
                    }

                # Originals — dedup across source models
                for it in (ctx.get("original_citations") or {}).get("items", []) or []:
                    pos = it.get("position")
                    if it.get("dimensions_id"):
                        pub_ids.add(it["dimensions_id"])
                    key = (arxiv, ctx_idx, pos)
                    if key in original_seen:
                        continue
                    original_seen.add(key)
                    original_records.append({
                        "focal_arxiv_id":  arxiv,
                        "context_index":   ctx_idx,
                        "position":        pos,
                        "cite_key":        it.get("cite_key"),
                        "cited_dim_id":    it.get("dimensions_id"),
                        "cited_doi_csv":   it.get("ref_doi") or "",
                        "cited_year_csv":  it.get("ref_year") or "",
                        "cited_venue_csv": it.get("ref_venue") or "",
                        "cited_title_csv": it.get("ref_title") or "",
                        "match_method":    it.get("match_method"),
                        "doi_agrees":      it.get("doi_agrees_with_match"),
                        "year_agrees":     it.get("year_agrees_with_match"),
                    })

                # Recommended — per source model
                ms  = ctx.get("motivation_self")          or {}
                mjf = ctx.get("motivation_judge_filled")  or {}
                for it in (ctx.get("llm_generated_citations") or {}).get("items", []) or []:
                    if it.get("dimensions_id"):
                        pub_ids.add(it["dimensions_id"])
                    llm_records[sm].append({
                        "source_model":              sm,
                        "focal_arxiv_id":            arxiv,
                        "context_index":             ctx_idx,
                        "position":                  it.get("position"),
                        "rec_title":                 it.get("rec_title"),
                        "rec_authors_raw":           it.get("rec_authors"),
                        "cited_dim_id":              it.get("dimensions_id"),
                        "cited_doi_csv":             it.get("rec_doi") or "",
                        "cited_year_csv":            it.get("rec_year"),
                        "cited_venue_csv":           it.get("rec_venue") or "",
                        "match_method":              it.get("match_method"),
                        "doi_agrees":                it.get("doi_agrees_with_match"),
                        "year_agrees":               it.get("year_agrees_with_match"),
                        "motivation_self":           ms.get("motivation")  if isinstance(ms,  dict) else None,
                        "confidence_self":           ms.get("confidence")  if isinstance(ms,  dict) else None,
                        "motivation_judge_filled":   mjf.get("motivation") if isinstance(mjf, dict) else None,
                        "confidence_judge_filled":   mjf.get("confidence") if isinstance(mjf, dict) else None,
                    })

    return pub_ids, paper_meta, ctx_meta, original_records, llm_records


# ============================================================
# STEP 2-3 — BQ: pubs_authors
# ============================================================

def query_pub_authors(pub_ids: set) -> pd.DataFrame:
    client = bigquery.Client(project=BQ_PROJECT, location=BQ_LOCATION)

    if os.getenv("SKIP_BQ_PUBS", "0") == "1":
        print(f"SKIP_BQ_PUBS=1: re-downloading existing `{BQ_PUB_AUTHORS_TABLE}`")
        t0 = time.time()
        df = client.query(
            f"SELECT * FROM `{BQ_PUB_AUTHORS_TABLE}`"
        ).result().to_dataframe(create_bqstorage_client=False)
        print(f"  download done in {time.time()-t0:.1f}s, {len(df):,} rows")
        return df

    # Ensure dataset exists
    ds_ref = bigquery.Dataset(f"{BQ_PROJECT}.{BQ_DATASET}")
    ds_ref.location = BQ_LOCATION
    try:
        client.get_dataset(ds_ref)
    except Exception:
        client.create_dataset(ds_ref, exists_ok=True)

    # Upload pub_ids
    df_in = pd.DataFrame({"pub_id": sorted(pub_ids)})
    schema = [bigquery.SchemaField("pub_id", "STRING")]
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    print(f"Uploading {len(df_in):,} pub_ids -> {BQ_PUB_LOOKUP_TABLE}")
    t0 = time.time()
    client.load_table_from_dataframe(df_in, BQ_PUB_LOOKUP_TABLE, job_config=job_config).result()
    print(f"  upload done in {time.time()-t0:.1f}s")

    sql = f"""
CREATE OR REPLACE TABLE `{BQ_PUB_AUTHORS_TABLE}` AS
WITH ids AS (SELECT pub_id FROM `{BQ_PUB_LOOKUP_TABLE}`)
SELECT
  p.id                                                                    AS dim_id,
  p.doi                                                                   AS doi,
  p.arxiv_id                                                              AS arxiv_id,
  p.year                                                                  AS year,
  COALESCE(p.conference.name, p.source.title)                             AS venue,
  LOWER(p.type)                                                           AS type,
  ARRAY_LENGTH(p.authors)                                                 AS teamsize,
  p.citations_count                                                       AS citations_count,

  p.authors[SAFE_OFFSET(0)].researcher_id                                 AS first_author_researcher_id,
  p.authors[SAFE_OFFSET(0)].first_name                                    AS first_author_first_name,
  p.authors[SAFE_OFFSET(0)].last_name                                     AS first_author_last_name,
  p.authors[SAFE_OFFSET(0)].orcid                                         AS first_author_orcid,

  p.authors[SAFE_OFFSET(ARRAY_LENGTH(p.authors) - 1)].researcher_id       AS last_author_researcher_id,
  p.authors[SAFE_OFFSET(ARRAY_LENGTH(p.authors) - 1)].first_name          AS last_author_first_name,
  p.authors[SAFE_OFFSET(ARRAY_LENGTH(p.authors) - 1)].last_name           AS last_author_last_name,
  p.authors[SAFE_OFFSET(ARRAY_LENGTH(p.authors) - 1)].orcid               AS last_author_orcid
FROM `{DIMENSIONS_PUBS_TABLE}` p
INNER JOIN ids i ON i.pub_id = p.id
"""
    print("Running pub_authors SQL ...")
    t0 = time.time()
    job = client.query(sql)
    job.result()
    bb = job.total_bytes_billed or 0
    print(f"  done in {time.time()-t0:.1f}s, billed {bb/1e9:.2f} GB (~${bb/1e12*USD_PER_TB:.3f})")

    print("Downloading pub_authors ...")
    t0 = time.time()
    df = client.query(
        f"SELECT * FROM `{BQ_PUB_AUTHORS_TABLE}`"
    ).result().to_dataframe(create_bqstorage_client=False)
    print(f"  download done in {time.time()-t0:.1f}s, {len(df):,} rows")
    return df


# ============================================================
# STEP 4-6 — BQ: researchers_metadata
# ============================================================

def query_researcher_metadata(researcher_ids: set) -> pd.DataFrame:
    client = bigquery.Client(project=BQ_PROJECT, location=BQ_LOCATION)

    if os.getenv("SKIP_BQ_RESEARCHERS", "0") == "1":
        print(f"SKIP_BQ_RESEARCHERS=1: re-downloading existing `{BQ_RESEARCHERS_OUT_TABLE}`")
        t0 = time.time()
        df = client.query(
            f"SELECT * FROM `{BQ_RESEARCHERS_OUT_TABLE}`"
        ).result().to_dataframe(create_bqstorage_client=False)
        print(f"  download done in {time.time()-t0:.1f}s, {len(df):,} rows")
        return df

    df_in = pd.DataFrame({"researcher_id": sorted(researcher_ids)})
    schema = [bigquery.SchemaField("researcher_id", "STRING")]
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    print(f"Uploading {len(df_in):,} researcher_ids -> {BQ_RESEARCHERS_LOOKUP_TABLE}")
    t0 = time.time()
    client.load_table_from_dataframe(
        df_in, BQ_RESEARCHERS_LOOKUP_TABLE, job_config=job_config
    ).result()
    print(f"  upload done in {time.time()-t0:.1f}s")

    sql = f"""
CREATE OR REPLACE TABLE `{BQ_RESEARCHERS_OUT_TABLE}` AS
WITH
  ids AS (SELECT researcher_id FROM `{BQ_RESEARCHERS_LOOKUP_TABLE}`),

  -- Per (researcher, publication) facts; restrict to publications of researchers
  -- we care about (CROSS JOIN UNNEST then INNER JOIN ids) and to year<=cutoff.
  pub_facts AS (
    SELECT
      r          AS researcher_id,
      p.id       AS pub_id,
      p.year     AS pub_year,
      p.research_org_countries,
      (SELECT COUNT(*) FROM UNNEST(p.citations) c
        WHERE c.year < {CITATION_YEAR_CUTOFF}) AS cites_received_before_cutoff
    FROM `{DIMENSIONS_PUBS_TABLE}` p
    CROSS JOIN UNNEST(p.researcher_ids) AS r
    INNER JOIN ids i ON i.researcher_id = r
    WHERE p.year <= {YEAR_CUTOFF}
  ),

  -- Researcher-level aggregations
  agg AS (
    SELECT
      researcher_id,
      MIN(pub_year)                            AS first_publication_year,
      MAX(pub_year)                            AS last_publication_year,
      COUNT(DISTINCT pub_id)                   AS total_publications_until_cutoff,
      SUM(cites_received_before_cutoff)        AS total_citations_received_before_cutoff
    FROM pub_facts
    GROUP BY researcher_id
  ),

  -- First-publication affiliation country.  Prefer rows with non-empty
  -- research_org_countries; otherwise just take the earliest pub.
  first_pub_ranked AS (
    SELECT
      researcher_id,
      research_org_countries[SAFE_OFFSET(0)] AS first_country_candidate,
      ROW_NUMBER() OVER (
        PARTITION BY researcher_id
        ORDER BY
          CASE WHEN ARRAY_LENGTH(research_org_countries) > 0 THEN 0 ELSE 1 END,
          pub_year ASC,
          pub_id   ASC
      ) AS rn
    FROM pub_facts
  ),
  first_country AS (
    SELECT researcher_id,
           NULLIF(first_country_candidate, '') AS first_affiliation_country
    FROM first_pub_ranked WHERE rn = 1
  ),

  -- Researcher-level metadata from the dedicated researchers table
  r_info AS (
    SELECT
      rr.id                            AS researcher_id,
      rr.first_name,
      rr.last_name,
      rr.initials,
      rr.orcid_ids,
      rr.current_research_org,
      rr.total_publications            AS rr_total_publications_alltime,
      rr.first_publication_year        AS rr_first_publication_year_alltime,
      rr.last_publication_year         AS rr_last_publication_year_alltime
    FROM `{DIMENSIONS_RESEARCHERS_TABLE}` rr
    INNER JOIN ids i ON i.researcher_id = rr.id
  )

SELECT
  ri.researcher_id,
  ri.first_name,
  ri.last_name,
  ri.initials,
  ri.orcid_ids,
  ri.current_research_org,

  -- All-time figures from researchers table (for cross-check)
  ri.rr_total_publications_alltime,
  ri.rr_first_publication_year_alltime,
  ri.rr_last_publication_year_alltime,

  -- Computed-from-publications, year-cutoff applied
  ag.first_publication_year,
  ag.last_publication_year,
  ag.total_publications_until_cutoff       AS total_publications_until_2024,
  ag.total_citations_received_before_cutoff AS total_citations_received_before_2025,
  ({CAREER_REF_YEAR} - ag.first_publication_year) AS career_age_at_2025,

  fc.first_affiliation_country
FROM r_info ri
LEFT JOIN agg           ag ON ag.researcher_id = ri.researcher_id
LEFT JOIN first_country fc ON fc.researcher_id = ri.researcher_id
"""
    print("Running researchers_metadata SQL ...")
    t0 = time.time()
    job = client.query(sql)
    job.result()
    bb = job.total_bytes_billed or 0
    print(f"  done in {time.time()-t0:.1f}s, billed {bb/1e9:.2f} GB (~${bb/1e12*USD_PER_TB:.3f})")

    print("Downloading researcher_metadata ...")
    t0 = time.time()
    df = client.query(
        f"SELECT * FROM `{BQ_RESEARCHERS_OUT_TABLE}`"
    ).result().to_dataframe(create_bqstorage_client=False)
    print(f"  download done in {time.time()-t0:.1f}s, {len(df):,} rows")
    return df


# ============================================================
# STEP 7 — BUILD DYAD TABLES
# ============================================================

def _author_cols(pub_row: pd.Series | None, prefix: str) -> dict:
    """Pull first/last author columns out of a single pub_authors row.
    pub_row may be None when the cited dim_id wasn't found in publications."""
    if pub_row is None:
        return {
            f"{prefix}_first_author_researcher_id": None,
            f"{prefix}_first_author_first_name":   None,
            f"{prefix}_first_author_last_name":    None,
            f"{prefix}_first_author_orcid":        None,
            f"{prefix}_last_author_researcher_id": None,
            f"{prefix}_last_author_first_name":    None,
            f"{prefix}_last_author_last_name":     None,
            f"{prefix}_last_author_orcid":         None,
            f"{prefix}_teamsize":                  None,
            f"{prefix}_citations_count":           None,
            f"{prefix}_type":                      None,
        }

    def _strip(v):
        # Treat NaN scalars as None; pass everything else through.
        if isinstance(v, float) and v != v:
            return None
        return v

    def _orcid(v):
        # orcid is a BQ REPEATED column -> numpy array in pandas. Normalize to
        # a plain list[str] or None. Avoid truthiness on arrays.
        if v is None:
            return None
        if isinstance(v, float) and v != v:
            return None
        try:
            n = len(v)
        except TypeError:
            # Scalar string
            if isinstance(v, str) and v == "":
                return None
            return [v]
        if n == 0:
            return None
        return [str(x) for x in v if x is not None and str(x) != ""]

    return {
        f"{prefix}_first_author_researcher_id": _strip(pub_row.get("first_author_researcher_id")),
        f"{prefix}_first_author_first_name":   _strip(pub_row.get("first_author_first_name")),
        f"{prefix}_first_author_last_name":    _strip(pub_row.get("first_author_last_name")),
        f"{prefix}_first_author_orcid":        _orcid(pub_row.get("first_author_orcid")),
        f"{prefix}_last_author_researcher_id": _strip(pub_row.get("last_author_researcher_id")),
        f"{prefix}_last_author_first_name":    _strip(pub_row.get("last_author_first_name")),
        f"{prefix}_last_author_last_name":     _strip(pub_row.get("last_author_last_name")),
        f"{prefix}_last_author_orcid":         _orcid(pub_row.get("last_author_orcid")),
        f"{prefix}_teamsize":                  int(pub_row.get("teamsize")) if pd.notna(pub_row.get("teamsize")) else None,
        f"{prefix}_citations_count":           int(pub_row.get("citations_count")) if pd.notna(pub_row.get("citations_count")) else None,
        f"{prefix}_type":                      _strip(pub_row.get("type")),
    }


def build_dyad_tables(pub_authors_df: pd.DataFrame,
                      paper_meta: dict,
                      ctx_meta:   dict,
                      original_records: list[dict],
                      llm_records: dict[str, list[dict]]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    pub_authors_idx = pub_authors_df.set_index("dim_id")

    def get_pub(dim_id):
        if dim_id is None or (isinstance(dim_id, float) and dim_id != dim_id):
            return None
        try:
            row = pub_authors_idx.loc[dim_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row
        except KeyError:
            return None

    def enrich(records: list[dict], citation_kind: str) -> pd.DataFrame:
        out = []
        for rec in records:
            arxiv     = rec["focal_arxiv_id"]
            pm        = paper_meta.get(arxiv, {})
            ctx_key   = (arxiv, rec["context_index"])
            cm        = ctx_meta.get(ctx_key, {})

            focal_pub = get_pub(pm.get("focal_dim_id"))
            cited_pub = get_pub(rec.get("cited_dim_id"))

            row = {
                "citation_kind":            citation_kind,
                "focal_arxiv_id":           arxiv,
                "focal_dim_id":             safe_str(pm.get("focal_dim_id")),
                "focal_preprint_dim_id":    safe_str(pm.get("focal_preprint_dim_id")),
                "focal_doi":                safe_str(pm.get("focal_doi")),
                "focal_title":              safe_str(pm.get("focal_title")),
                "focal_year":               safe_int(pm.get("focal_year")),
                "focal_venue":              safe_str(pm.get("focal_venue")),
                "focal_preprint_date":      safe_str(pm.get("focal_preprint_date")),
                **_author_cols(focal_pub, "focal"),

                "context_index":             safe_int(rec.get("context_index")),
                "section":                   safe_str(cm.get("section")),
                "position":                  safe_int(rec.get("position")),
                "num_citations_required":    safe_int(cm.get("num_citations_required")),

                "cited_dim_id":    safe_str(rec.get("cited_dim_id")),
                "cited_arxiv_id":  safe_str(cited_pub.get("arxiv_id") if cited_pub is not None else None),
                "cited_doi":       safe_str(
                                       (cited_pub.get("doi") if cited_pub is not None else None)
                                       or rec.get("cited_doi_csv")
                                   ),
                "cited_year":      safe_int(
                                       cited_pub.get("year") if cited_pub is not None else rec.get("cited_year_csv")
                                   ),
                "cited_venue":     safe_str(
                                       cited_pub.get("venue") if cited_pub is not None else rec.get("cited_venue_csv")
                                   ),
                **_author_cols(cited_pub, "cited"),

                "match_method":                  safe_str(rec.get("match_method")),
                "doi_agrees_with_match":         rec.get("doi_agrees"),
                "year_agrees_with_match":        rec.get("year_agrees"),

                "motivation_judge_original":     safe_str(cm.get("motivation_judge_original")),
                "confidence_judge_original":     safe_str(cm.get("confidence_judge_original")),
            }

            if citation_kind == "original":
                row["cite_key"]      = safe_str(rec.get("cite_key"))
                row["ref_title_csv"] = safe_str(rec.get("cited_title_csv"))
            else:
                row["source_model"]              = safe_str(rec.get("source_model"))
                row["rec_title"]                 = safe_str(rec.get("rec_title"))
                row["rec_authors_raw"]           = safe_str(rec.get("rec_authors_raw"))
                row["motivation_self"]           = safe_str(rec.get("motivation_self"))
                row["confidence_self"]           = safe_str(rec.get("confidence_self"))
                row["motivation_judge_filled"]   = safe_str(rec.get("motivation_judge_filled"))
                row["confidence_judge_filled"]   = safe_str(rec.get("confidence_judge_filled"))

            out.append(row)
        return pd.DataFrame(out)

    # --- ORIGINAL dyads (shared across all source models) ---
    print(f"Building original dyads ({len(original_records):,} records)...")
    orig_df = enrich(original_records, "original")
    out_path = os.path.join(OUT_DIR, "original_dyads.parquet")
    orig_df.to_parquet(out_path, index=False, compression="zstd")
    n_matched = int(orig_df["cited_dim_id"].notna().sum())
    n_cited_first_with_rid = int(orig_df["cited_first_author_researcher_id"].notna().sum())
    n_cited_last_with_rid  = int(orig_df["cited_last_author_researcher_id"].notna().sum())
    print(f"  wrote {out_path}  rows={len(orig_df):,}  "
          f"cited_matched={n_matched:,}  cited_first_with_rid={n_cited_first_with_rid:,}  "
          f"cited_last_with_rid={n_cited_last_with_rid:,}")

    # --- LLM dyads per source model ---
    for sm, recs in sorted(llm_records.items()):
        sm_slug = sanitize_model_slug(sm)
        print(f"Building LLM dyads for {sm} ({len(recs):,} records)...")
        df = enrich(recs, "recommended")
        out_path = os.path.join(OUT_DIR, f"llm_generated__{sm_slug}.parquet")
        df.to_parquet(out_path, index=False, compression="zstd")
        n_matched = int(df["cited_dim_id"].notna().sum())
        print(f"  wrote {out_path}  rows={len(df):,}  cited_matched={n_matched:,} "
              f"({n_matched/max(len(df),1)*100:.2f}%)")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("Step 04 — author dyads + researcher metadata")
    print(f"  judge dointitle dir:  {JUDGE_OUT_DIR}")
    print(f"  out dir:              {OUT_DIR}")
    print(f"  BQ:                   {BQ_PROJECT}.{BQ_DATASET}")
    print(f"  year cutoff:          publications<={YEAR_CUTOFF}, citations<{CITATION_YEAR_CUTOFF}")
    print("=" * 72)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n[1/4] Walking JSONs...")
    t0 = time.time()
    pub_ids, paper_meta, ctx_meta, original_records, llm_records = walk_jsons()
    print(f"  walk done in {time.time()-t0:.1f}s")
    print(f"  unique pub_ids:                {len(pub_ids):,}")
    print(f"  unique focal papers:           {len(paper_meta):,}")
    print(f"  unique (paper,ctx) contexts:   {len(ctx_meta):,}")
    print(f"  original citation rows:        {len(original_records):,}")
    n_llm = sum(len(v) for v in llm_records.values())
    print(f"  LLM citation rows (all 6 sm):  {n_llm:,}")

    print("\n[2/4] Querying pub_authors from BigQuery...")
    pub_authors_df = query_pub_authors(pub_ids)
    pa_path = os.path.join(OUT_DIR, "pub_authors.parquet")
    pub_authors_df.to_parquet(pa_path, index=False, compression="zstd")
    print(f"  saved {pa_path}  ({len(pub_authors_df):,} rows, "
          f"{int(pub_authors_df['first_author_researcher_id'].notna().sum()):,} have first-author rid, "
          f"{int(pub_authors_df['last_author_researcher_id'].notna().sum()):,} have last-author rid)")

    # Collect researcher_ids
    researcher_ids: set = set()
    for col in ("first_author_researcher_id", "last_author_researcher_id"):
        researcher_ids.update(
            x for x in pub_authors_df[col].dropna().unique()
            if isinstance(x, str) and x.strip()
        )
    print(f"  unique researcher_ids to look up: {len(researcher_ids):,}")

    print("\n[3/4] Querying researcher_metadata from BigQuery...")
    researcher_meta_df = query_researcher_metadata(researcher_ids)
    rm_path = os.path.join(OUT_DIR, "researcher_metadata.parquet")
    researcher_meta_df.to_parquet(rm_path, index=False, compression="zstd")
    print(f"  saved {rm_path}  ({len(researcher_meta_df):,} rows)")

    print("\n[4/4] Building dyad parquet files...")
    build_dyad_tables(pub_authors_df, paper_meta, ctx_meta, original_records, llm_records)

    print("\nDone.")


if __name__ == "__main__":
    main()
