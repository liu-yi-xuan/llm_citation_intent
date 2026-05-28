"""
Step 05 — Coauthorship edge list for every researcher in
researcher_metadata.parquet, restricted to publications in 2015-2024.

Source set: the 89,253 researcher_ids already staged in BQ as
  <project>.<dataset>.researchers_to_lookup
  (populated by Step 04).

Algorithm (one SQL job on BigQuery):
  1. For every publication with year in [2015, 2025), build the
     DEDUPED array of non-null researcher_ids that appear as authors.
  2. Drop pubs with fewer than 2 disambiguated authors, and drop pubs
     that don't contain at least one of our researchers.
  3. Within each surviving pub, enumerate UNORDERED pairs (a, b) with a < b
     so each undirected edge appears exactly once per pub.
  4. KEEP EVERY within-pub pair, including pairs where NEITHER endpoint
     is in our researcher set.  (The previous "ego-graph" version dropped
     those; this version keeps them so BFS distances see the full
     coauthor connectivity that exists inside our papers.)
  5. GROUP BY (a, b) and count rows -> Weights = # joint publications.

Resulting table has EXACTLY three columns (Source, Target, Weights),
representing the UNDIRECTED weighted coauthorship graph among researchers
appearing in any pub that contains at least one of our 89,253 researchers:
   - Source : researcher_id, always lex-smaller of the two endpoints
   - Target : researcher_id, always lex-greater of the two endpoints
   - Weights: int64, # distinct joint publications in the year window

Each undirected edge appears EXACTLY ONCE.  Endpoints may or may not be
in our researcher set.  Load directly as `nx.Graph()` (undirected, single
edge per pair).

Caveat: this graph still excludes coauthor edges from pubs that contain
NONE of our researchers.  For a fully global 2015-2024 coauthor graph
(option G1), drop the `pubs_with_our` filter from the SQL.

Outputs:
  BigQuery: <project>.<dataset>.coauthor_edges_2015_2024
  Local:    data/author_dyads_judge_gemini/coauthor_edges_2015_2024.parquet

Set SKIP_BQ_EDGES=1 to skip the SQL job and just re-download the cached table.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
from google.cloud import bigquery


# ============================================================
# CONFIG
# ============================================================

# Configure via env (or config.yml — see config.example.yml). Anonymized defaults.
BQ_PROJECT           = os.environ.get("BQ_PROJECT", "your-gcp-project")
BQ_DATASET           = os.environ.get("BQ_DATASET", "your_dataset")
BQ_RESEARCHERS_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.researchers_to_lookup"
BQ_EDGES_TABLE       = f"{BQ_PROJECT}.{BQ_DATASET}.coauthor_edges_2015_2024"
BQ_LOCATION          = "US"

DIMENSIONS_PUBS_TABLE = "dimensions-ai.data_analytics.publications"

YEAR_LO = 2015   # inclusive
YEAR_HI = 2025   # exclusive

# Drop pubs with more disambiguated coauthors than this.  Mega-team papers
# (HEP/medical/genomics with 100+ authors) contribute C(N,2) edges each
# and dominate the edge count without producing meaningful 1-on-1
# collaboration links.  25 keeps essentially every NLP paper.
MAX_TEAMSIZE = 25

OUT_DIR     = "data/author_dyads_judge_gemini"
OUT_PARQUET = os.path.join(OUT_DIR, "coauthor_edges_2015_2024.parquet")

USD_PER_TB  = 6.25


def main() -> None:
    print("=" * 72)
    print(f"Step 05 — coauthor edge list, year window [{YEAR_LO}, {YEAR_HI})")
    print(f"  max teamsize per pub: {MAX_TEAMSIZE}")
    print(f"  source rids table:  {BQ_RESEARCHERS_TABLE}")
    print(f"  output BQ table:    {BQ_EDGES_TABLE}")
    print(f"  output parquet:     {OUT_PARQUET}")
    print("=" * 72)

    client = bigquery.Client(project=BQ_PROJECT, location=BQ_LOCATION)

    if os.getenv("SKIP_BQ_EDGES", "0") == "1":
        print(f"\nSKIP_BQ_EDGES=1: re-downloading existing `{BQ_EDGES_TABLE}`")
        t0 = time.time()
        df = client.query(
            f"SELECT * FROM `{BQ_EDGES_TABLE}`"
        ).result().to_dataframe(create_bqstorage_client=False)
        print(f"  download done in {time.time()-t0:.1f}s, {len(df):,} edges")
    else:
        # Verify source rids table exists
        try:
            client.get_table(BQ_RESEARCHERS_TABLE)
        except Exception as e:
            raise SystemExit(
                f"Source rids table not found: {BQ_RESEARCHERS_TABLE}\n"
                f"Run Step 04 first to populate it."
            ) from e

        sql = f"""
CREATE OR REPLACE TABLE `{BQ_EDGES_TABLE}` AS
WITH
  our_rids AS (
    SELECT researcher_id FROM `{BQ_RESEARCHERS_TABLE}`
  ),
  pub_rids AS (
    -- For every pub in the window, the deduped list of disambiguated authors
    SELECT
      p.id AS pub_id,
      ARRAY(
        SELECT DISTINCT a.researcher_id
        FROM UNNEST(p.authors) a
        WHERE a.researcher_id IS NOT NULL
          AND a.researcher_id != ''
      ) AS rids
    FROM `{DIMENSIONS_PUBS_TABLE}` p
    WHERE p.year >= {YEAR_LO} AND p.year < {YEAR_HI}
  ),
  -- pub_ids that contain at least one of our researchers
  pubs_with_our AS (
    SELECT DISTINCT pr.pub_id
    FROM pub_rids pr
    CROSS JOIN UNNEST(pr.rids) AS r
    INNER JOIN our_rids ON our_rids.researcher_id = r
  ),
  pubs_of_interest AS (
    SELECT pr.pub_id, pr.rids
    FROM pub_rids pr
    INNER JOIN pubs_with_our pwo ON pwo.pub_id = pr.pub_id
    WHERE ARRAY_LENGTH(pr.rids) >= 2
      AND ARRAY_LENGTH(pr.rids) <= {MAX_TEAMSIZE}
  ),
  pairs AS (
    -- All unordered pairs (a < b) within each pub of interest.
    -- We keep every pair, including ones where neither endpoint
    -- is in our researcher set (e.g., the B-C edge in pub authors
    -- A,B,C where A is the only "in-set" author).
    SELECT a AS src, b AS tgt
    FROM pubs_of_interest poi
    CROSS JOIN UNNEST(poi.rids) AS a
    CROSS JOIN UNNEST(poi.rids) AS b
    WHERE a < b
  )
SELECT
  src AS Source,
  tgt AS Target,
  COUNT(*) AS Weights
FROM pairs
GROUP BY src, tgt
"""
        print("\nRunning edge-list SQL ...")
        t0 = time.time()
        job = client.query(sql)
        job.result()
        bb = job.total_bytes_billed or 0
        print(f"  done in {time.time()-t0:.1f}s, "
              f"billed {bb/1e9:.2f} GB (~${bb/1e12*USD_PER_TB:.3f})")

        print("Downloading edge table ...")
        t0 = time.time()
        df = client.query(
            f"SELECT * FROM `{BQ_EDGES_TABLE}`"
        ).result().to_dataframe(create_bqstorage_client=False)
        print(f"  download done in {time.time()-t0:.1f}s, {len(df):,} edges")

    # Save parquet
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False, compression="zstd")
    print(f"\nSaved {OUT_PARQUET}")

    # Sanity summary
    print("\n" + "=" * 60)
    print("SANITY SUMMARY")
    print("=" * 60)
    print(f"  total undirected edges:            {len(df):,}")
    print(f"  unique nodes in Source col:        {df['Source'].nunique():,}")
    print(f"  unique nodes in Target col:        {df['Target'].nunique():,}")
    print(f"  unique nodes overall:              "
          f"{pd.concat([df['Source'], df['Target']]).nunique():,}")

    try:
        rm_path = os.path.join(OUT_DIR, "researcher_metadata.parquet")
        rids_in_set = set(pd.read_parquet(rm_path)["researcher_id"])
        s_in = df["Source"].isin(rids_in_set)
        t_in = df["Target"].isin(rids_in_set)
        both    = int((s_in & t_in).sum())
        one_src = int((s_in & ~t_in).sum())
        one_tgt = int((~s_in & t_in).sum())
        print(f"  both endpoints in our set:         {both:>10,}  "
              f"({both/len(df)*100:.2f}%)")
        print(f"  only Source in our set:            {one_src:>10,}  "
              f"({one_src/len(df)*100:.2f}%)")
        print(f"  only Target in our set:            {one_tgt:>10,}  "
              f"({one_tgt/len(df)*100:.2f}%)")
        print(f"  neither endpoint in our set:       "
              f"{int((~s_in & ~t_in).sum()):>10,}  "
              f"({int((~s_in & ~t_in).sum())/len(df)*100:.2f}%)")
        # Verify canonical ordering (Source < Target always)
        bad = int((df["Source"] >= df["Target"]).sum())
        print(f"  rows where Source >= Target (sanity check, should be 0): {bad}")
    except FileNotFoundError:
        pass

    w = df["Weights"]
    print(f"\n  weight distribution:")
    print(f"    min   = {int(w.min())}")
    print(f"    p25   = {int(w.quantile(0.25))}")
    print(f"    p50   = {int(w.quantile(0.50))}")
    print(f"    p75   = {int(w.quantile(0.75))}")
    print(f"    p90   = {int(w.quantile(0.90))}")
    print(f"    p99   = {int(w.quantile(0.99))}")
    print(f"    max   = {int(w.max())}")
    print(f"    sum   = {int(w.sum()):,}  (= total directed coauthor-paper-incidences)")

    print("\nDone.")


if __name__ == "__main__":
    main()
