"""
Step 06 — Shortest-path distances between focal-paper authors and cited-paper
authors over the 2015-2024 undirected unweighted coauthorship graph.

NOTE on library choice
----------------------
The user originally requested graph-tool, but graph-tool is conda-forge-only
and not pip-installable.  We use **networkit** instead, which is:
  * pip-installable (already in the env),
  * C++-backed (same speed class as graph-tool),
  * algorithmically equivalent for unweighted-undirected BFS.

If you switch to graph-tool later, only the `_GraphBackend` shim below needs
to change (add_edge / bfs_distances / num_vertices).

Pipeline
--------
1.  Walk `coauthor_edges_2015_2024.parquet`, assign each unique researcher_id
    a sequential int32 node id.  Save mapping to `node_id_mapping.parquet`.
2.  Build a networkit Graph (undirected, unweighted) from the edge list.
3.  Load all 7 dyad parquets (`original_dyads.parquet` +
    `llm_generated__*.parquet`), map the four author columns
    (focal_first/last, cited_first/last) to int node ids.
4.  Collect the UNION of focal-source vertices across all 7 files
    (typically ~2-3k unique).
5.  For each source S in that union, run BFS ONCE -> dist array -> update
    every row in every dyad file where focal_first==S or focal_last==S.
    This reuse across 7 files avoids 7x redundant BFS work.
6.  Save each dyad parquet as `..._with_distances.csv` with four new columns:
       dist_focal_first_to_cited_first
       dist_focal_first_to_cited_last
       dist_focal_last_to_cited_first
       dist_focal_last_to_cited_last

Distance conventions
--------------------
   pd.NA   either endpoint lacks a researcher_id, OR the rid is not present
           in the coauthor graph (e.g., authors of 2025 papers not yet
           ingested by Dimensions).
   -1      both endpoints are in the graph but no path exists between them
           (disconnected components).
   0       both endpoints are the same person.
   integer >0  BFS hop count.

Parallelism
-----------
   WORKERS=N (env var, default 1).  When N>1 we use multiprocessing where
   each worker rebuilds the graph in its initializer (~30-60s overhead).

Progress / ETA
--------------
   tqdm progress bar with rate.  After the first 50 BFS calls, prints an
   ETA estimate based on observed throughput.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import networkit as nk
    nk.setNumberOfThreads(1)  # let our outer-level mp pool drive parallelism
except ImportError:
    print("ERROR: networkit is not installed.")
    sys.exit(1)


# ============================================================
# CONFIG
# ============================================================

D = "data/author_dyads_judge_gemini"
EDGES_PARQUET    = f"{D}/coauthor_edges_2015_2024.parquet"
NODE_MAP_PARQUET = f"{D}/node_id_mapping.parquet"

DYAD_FILES = [f"{D}/original_dyads.parquet"] + sorted(
    str(p) for p in Path(D).glob("llm_generated__*.parquet"))

WORKERS = int(os.getenv("WORKERS", "1"))

# Distance-column names
DIST_COLS = {
    ("ff", "cf"): "dist_focal_first_to_cited_first",
    ("ff", "cl"): "dist_focal_first_to_cited_last",
    ("fl", "cf"): "dist_focal_last_to_cited_first",
    ("fl", "cl"): "dist_focal_last_to_cited_last",
}

RID_COL = {
    "ff": "focal_first_author_researcher_id",
    "fl": "focal_last_author_researcher_id",
    "cf": "cited_first_author_researcher_id",
    "cl": "cited_last_author_researcher_id",
}

# networkit BFS returns 1e308 (~max double) for unreachable.  We map to -1.
UNREACHABLE_THRESHOLD = 1e15


# ============================================================
# GRAPH BUILD
# ============================================================

def build_node_mapping(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Assign sequential int32 node ids to every unique researcher_id appearing
    in Source/Target of the edge list.  Save and return the mapping."""
    nodes = pd.unique(pd.concat([edges_df["Source"], edges_df["Target"]], ignore_index=True))
    mapping = pd.DataFrame({
        "node_id": np.arange(len(nodes), dtype=np.int32),
        "researcher_id": nodes,
    })
    mapping.to_parquet(NODE_MAP_PARQUET, index=False, compression="zstd")
    print(f"  saved {NODE_MAP_PARQUET}  ({len(mapping):,} unique researcher_ids)")
    return mapping


def build_graph_from_edges(edges_df: pd.DataFrame, mapping: pd.DataFrame) -> "nk.Graph":
    """Build a networkit Graph (undirected, unweighted) from an edge dataframe
    of (Source, Target, Weights) where Source/Target are researcher_id strings.
    The Weights column is intentionally ignored — graph is unweighted per spec."""
    rid_to_int = dict(zip(mapping["researcher_id"].values,
                          mapping["node_id"].astype(np.int64).values))

    n_nodes = len(mapping)
    print(f"  initializing networkit Graph: n={n_nodes:,}, "
          f"m={len(edges_df):,} (this allocates ~{n_nodes*16/1e6:.0f} MB)")
    t0 = time.time()
    g = nk.Graph(n=n_nodes, weighted=False, directed=False)

    src = edges_df["Source"].map(rid_to_int).astype(np.int64).values
    tgt = edges_df["Target"].map(rid_to_int).astype(np.int64).values

    # Bulk add edges via Python loop — fastest reliable path on networkit 11.
    add_edge = g.addEdge
    for s, t in zip(src, tgt):
        add_edge(s, t)
    print(f"  built graph in {time.time()-t0:.1f}s: "
          f"|V|={g.numberOfNodes():,}  |E|={g.numberOfEdges():,}")
    return g


# ============================================================
# DYAD ENRICHMENT
# ============================================================

def load_and_index_dyads(rid_to_int: dict) -> list[tuple[str, pd.DataFrame]]:
    """Load each dyad parquet, add int-coded columns ff_int/fl_int/cf_int/cl_int,
    and init the 4 distance columns to pd.NA."""
    out = []
    for fp in DYAD_FILES:
        if not os.path.exists(fp):
            continue
        df = pd.read_parquet(fp)
        for short, col in RID_COL.items():
            df[f"{short}_int"] = (
                df[col].map(rid_to_int).astype("Int64")
            )
        for _, dcol in DIST_COLS.items():
            df[dcol] = pd.NA
        out.append((fp, df))
        print(f"  loaded {os.path.basename(fp):70s} rows={len(df):,}")
    return out


def collect_unique_sources(dyads: list[tuple[str, pd.DataFrame]]) -> list[int]:
    """Union of focal_first_int + focal_last_int across all 7 dyad files."""
    s = set()
    for _, df in dyads:
        for short in ("ff", "fl"):
            s.update(int(x) for x in df[f"{short}_int"].dropna().unique())
    return sorted(s)


def update_rows_for_source(df: pd.DataFrame, dist: np.ndarray, src: int) -> None:
    """Given a BFS distance array from `src`, fill the 4 distance columns
    for every row in df where focal_first==src or focal_last==src."""
    for focal_short in ("ff", "fl"):
        focal_col = f"{focal_short}_int"
        mask = (df[focal_col] == src).values
        if not mask.any():
            continue
        for cited_short in ("cf", "cl"):
            cited_col = f"{cited_short}_int"
            dist_col  = DIST_COLS[(focal_short, cited_short)]
            tgt_series = df.loc[mask, cited_col]
            valid = tgt_series.notna().values
            if not valid.any():
                continue
            tgt_int = tgt_series[valid].astype("int64").values
            d_raw = dist[tgt_int]
            d_int = np.where(d_raw > UNREACHABLE_THRESHOLD, -1, d_raw).astype("int64")
            row_idx = df.index[mask][valid]
            df.loc[row_idx, dist_col] = d_int


# ============================================================
# BFS DRIVERS (sequential and parallel)
# ============================================================

def bfs_sequential(g: "nk.Graph", sources: list[int],
                   dyads: list[tuple[str, pd.DataFrame]]) -> None:
    """Single-process BFS loop with tqdm + ETA print after first 50 sources."""
    t_start = time.time()
    pbar = tqdm(sources, desc="BFS", smoothing=0.05)
    ETA_AT = 50
    for i, src in enumerate(pbar):
        bfs = nk.distance.BFS(g, src, storePaths=False)
        bfs.run()
        dist = np.asarray(bfs.getDistances(), dtype=np.float64)
        for _, df in dyads:
            update_rows_for_source(df, dist, src)

        if i == ETA_AT:
            elapsed = time.time() - t_start
            rate = (ETA_AT + 1) / elapsed
            remaining = (len(sources) - ETA_AT - 1) / rate
            print(f"\n  [ETA from {ETA_AT+1} BFS calls in {elapsed:.1f}s "
                  f"= {rate:.2f}/s] projected remaining ~{remaining/60:.1f} min")


# ---- Parallel version (mp.Pool) ----

_W_G = None  # per-worker graph (set in worker_init)


def _worker_init(edges_path: str, map_path: str) -> None:
    global _W_G
    edges = pd.read_parquet(edges_path)
    mapping = pd.read_parquet(map_path)
    rid_to_int = dict(zip(mapping["researcher_id"].values,
                          mapping["node_id"].astype(np.int64).values))
    n = len(mapping)
    g = nk.Graph(n=n, weighted=False, directed=False)
    src = edges["Source"].map(rid_to_int).astype(np.int64).values
    tgt = edges["Target"].map(rid_to_int).astype(np.int64).values
    add_edge = g.addEdge
    for s, t in zip(src, tgt):
        add_edge(s, t)
    _W_G = g


def _worker_bfs(src: int) -> tuple[int, np.ndarray]:
    bfs = nk.distance.BFS(_W_G, src, storePaths=False)
    bfs.run()
    dist = np.asarray(bfs.getDistances(), dtype=np.float64)
    # Compress: clamp unreachable to -1 (sentinel)
    dist = np.where(dist > UNREACHABLE_THRESHOLD, -1, dist).astype(np.int64)
    return src, dist


def bfs_parallel(workers: int, sources: list[int],
                 dyads: list[tuple[str, pd.DataFrame]]) -> None:
    import multiprocessing as mp
    print(f"  using {workers} worker processes (each rebuilds graph in init)")
    t_start = time.time()
    with mp.Pool(processes=workers, initializer=_worker_init,
                 initargs=(EDGES_PARQUET, NODE_MAP_PARQUET)) as pool:
        # imap_unordered keeps the queue fed
        pbar = tqdm(pool.imap_unordered(_worker_bfs, sources, chunksize=4),
                    total=len(sources), desc="BFS", smoothing=0.05)
        ETA_AT = 50
        for i, (src, dist) in enumerate(pbar):
            # dist already has -1 for unreachable; cast back to float for the update helper
            dist_f = dist.astype(np.float64)
            # but update_rows_for_source compares > UNREACHABLE_THRESHOLD; here our -1
            # values are <0 so they pass through.  Convert -1 sentinels to a large
            # value temporarily so they trip the threshold and stay -1 after the
            # `np.where` inside update_rows_for_source.
            sentinel_mask = dist == -1
            dist_f[sentinel_mask] = UNREACHABLE_THRESHOLD * 2
            for _, df in dyads:
                update_rows_for_source(df, dist_f, src)
            if i == ETA_AT:
                elapsed = time.time() - t_start
                rate = (ETA_AT + 1) / elapsed
                remaining = (len(sources) - ETA_AT - 1) / rate
                print(f"\n  [ETA from {ETA_AT+1} BFS calls in {elapsed:.1f}s "
                      f"= {rate:.2f}/s] projected remaining ~{remaining/60:.1f} min")


# ============================================================
# OUTPUT
# ============================================================

def save_csvs(dyads: list[tuple[str, pd.DataFrame]]) -> None:
    print("\nSaving CSVs with distance columns appended ...")
    for fp, df in dyads:
        # Drop the per-row int helper columns; keep only the public schema + new distances.
        df = df.drop(columns=[c for c in df.columns
                              if c in {"ff_int", "fl_int", "cf_int", "cl_int"}])
        out_path = fp.replace(".parquet", "_with_distances.csv")
        df.to_csv(out_path, index=False)
        # Quick coverage stats
        n_rows = len(df)
        for _, dcol in DIST_COLS.items():
            n_resolved = int(df[dcol].notna().sum())
            n_unreachable = int((df[dcol] == -1).sum())
            n_with_path = n_resolved - n_unreachable
            print(f"  {os.path.basename(out_path):65s}  "
                  f"{dcol:38s}  resolved={n_resolved:>7,}/{n_rows:,} "
                  f"(unreachable={n_unreachable:,}, with_path={n_with_path:,})")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("Step 06 — author-dyad shortest-path distances (BFS, undirected, unweighted)")
    print(f"  graph: {EDGES_PARQUET}")
    print(f"  WORKERS={WORKERS}")
    print("=" * 72)

    print("\n[1/5] Loading edges ...")
    edges = pd.read_parquet(EDGES_PARQUET)
    print(f"  {len(edges):,} edges")

    print("\n[2/5] Building node mapping ...")
    mapping = build_node_mapping(edges)
    rid_to_int = dict(zip(mapping["researcher_id"].values,
                          mapping["node_id"].astype(np.int64).values))

    print("\n[3/5] Building networkit graph ...")
    g = build_graph_from_edges(edges, mapping)
    # Free edges memory before BFS loop
    del edges

    print("\n[4/5] Loading and indexing dyad files ...")
    dyads = load_and_index_dyads(rid_to_int)
    if not dyads:
        raise SystemExit("No dyad parquet files found.")

    sources = collect_unique_sources(dyads)
    print(f"\n  unique source vertices to BFS from: {len(sources):,}")
    if not sources:
        raise SystemExit("No source vertices found (no focal-author rids in any dyad file).")

    # Quick estimate from a small sample
    print(f"\n  estimating BFS rate from 5 sample sources ...")
    t0 = time.time()
    for s in sources[:5]:
        bfs = nk.distance.BFS(g, s, storePaths=False)
        bfs.run()
    sample_rate = 5 / (time.time() - t0)
    est_total_sec = len(sources) / max(sample_rate, 1e-6)
    if WORKERS > 1:
        est_total_sec /= max(WORKERS - 1, 1)  # rough; ignores startup cost
    print(f"  sample BFS rate: {sample_rate:.2f}/s")
    print(f"  estimated total BFS time (workers={WORKERS}): "
          f"~{est_total_sec/60:.1f} min")

    print("\n[5/5] Running BFS + updating dyad rows ...")
    t0 = time.time()
    if WORKERS <= 1:
        bfs_sequential(g, sources, dyads)
    else:
        bfs_parallel(WORKERS, sources, dyads)
    print(f"  BFS phase done in {(time.time()-t0)/60:.1f} min")

    save_csvs(dyads)
    print("\nDone.")


if __name__ == "__main__":
    main()
