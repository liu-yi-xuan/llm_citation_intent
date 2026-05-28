"""
Shared analysis utilities for the citation-intent study.

Everything the three RQ scripts need in common lives here:
  * global matplotlib style (de-framed axes, embeddable TrueType fonts)
  * the canonical model ordering, palettes, and short display names
  * data loaders for the `*_with_distances.csv` dyad files
  * a few small statistics helpers (significance stars, Cohen's d)

Data layout (configurable via the CITATION_DATA_DIR env var)
------------------------------------------------------------
Each `<DATA_DIR>/*_with_distances.csv` is one row per citation slot
(focal paper x context x position) with, among many columns:

  motivation_judge_original   judge label on the human sentence (ORIGINAL only)
  motivation_judge_filled     judge label on the LLM-filled sentence (LLMs)
  motivation_self             the source LLM's own self-label (LLMs)
  section, focal_venue, focal_year, focal_arxiv_id, context_index, position
  cited_dim_id, cited_year, cited_citations_count, cited_teamsize
  dist_focal_{first,last}_to_cited_{first,last}   four BFS hop distances

Distance encoding: NaN = endpoint missing from the graph; -1 = both present
but disconnected; >=0 = BFS hop count (0 = self-cite, 1 = direct coauthor).

Outputs go to <FIG_DIR> (default ./figures), as PNG (+ PDF/SVG where noted).
"""

from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# Point this at the directory holding the *_with_distances.csv files. Datasets
# are distributed separately from the code (see the repo README).
DATA_DIR = os.environ.get(
    "CITATION_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "author_dyads_judge_gemini"),
)
FIG_DIR = os.environ.get(
    "CITATION_FIG_DIR",
    os.path.join(os.path.dirname(__file__), "..", "figures"),
)
os.makedirs(FIG_DIR, exist_ok=True)

DIST_COLS = [
    "dist_focal_first_to_cited_first",
    "dist_focal_first_to_cited_last",
    "dist_focal_last_to_cited_first",
    "dist_focal_last_to_cited_last",
]

MOT_ORDER = ["supporting", "contrasting", "mentioning"]
# Macaron palette: warm=supporting, blue=contrasting, green=mentioning.
MOT_COLORS = {"supporting":  "#F2938C",
              "contrasting": "#8FBCDB",
              "mentioning":  "#A6D49B"}

# ORIGINAL (the human baseline) first, then the six LLMs.
ORDER_PREFERENCE = [
    "ORIGINAL",
    "deepseek-ai_DeepSeek-V3.2",
    "openai_gpt-5.1-chat",
    "meta-llama_llama-4-maverick",
    "google_gemini-2.0-flash-001",
    "Qwen_Qwen2.5-72B-Instruct",
    "anthropic_claude-3.5-haiku",
]

SHORT_NAME = {
    "ORIGINAL":                    "Original",
    "deepseek-ai_DeepSeek-V3.2":   "DeepSeek-V3.2",
    "openai_gpt-5.1-chat":         "GPT-5.1",
    "meta-llama_llama-4-maverick": "Llama-4-Maverick",
    "google_gemini-2.0-flash-001": "Gemini-2.0-Flash",
    "Qwen_Qwen2.5-72B-Instruct":   "Qwen2.5-72B",
    "anthropic_claude-3.5-haiku":  "Claude-3.5-Haiku",
}
# Compact tick labels used on the small multi-bar figures.
COMPACT_NAME = {
    "ORIGINAL": "Original",
    "deepseek-ai_DeepSeek-V3.2": "DeepSeek",
    "openai_gpt-5.1-chat": "GPT-5.1",
    "meta-llama_llama-4-maverick": "Llama-4",
    "google_gemini-2.0-flash-001": "Gemini-2.0",
    "Qwen_Qwen2.5-72B-Instruct": "Qwen-72B",
    "anthropic_claude-3.5-haiku": "Claude-3.5",
}

# Per-model background-band colours: ORIGINAL neutral grey, LLMs a consistent
# BuPu ramp so the same model reads the same colour across every figure.
_LLM_LABELS_ORDERED = [l for l in ORDER_PREFERENCE if l != "ORIGINAL"]
MODEL_BAND_COLORS = {"ORIGINAL": "#808080"}
_bupu = plt.get_cmap("BuPu")
for _i, _lbl in enumerate(_LLM_LABELS_ORDERED):
    MODEL_BAND_COLORS[_lbl] = _bupu(0.35 + 0.60 * _i / max(len(_LLM_LABELS_ORDERED) - 1, 1))
MODEL_BAND_ALPHA = 0.05

TOP_SECTIONS = ["Introduction", "Background", "Related Work", "Method",
                "Experiments", "Discussion", "Conclusion", "Limitations"]


def apply_style() -> None:
    """De-frame upper/left/right spines, horizontal dashed grid, embed fonts."""
    plt.rcParams["axes.spines.top"]   = False
    plt.rcParams["axes.spines.left"]  = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["ytick.left"]        = False
    plt.rcParams["axes.grid"]         = True
    plt.rcParams["axes.grid.which"]   = "major"
    plt.rcParams["axes.grid.axis"]    = "y"
    plt.rcParams["grid.linestyle"]    = "--"
    plt.rcParams["grid.alpha"]        = 0.3
    plt.rcParams["pdf.fonttype"]      = 42   # TrueType, editable in Illustrator
    plt.rcParams["ps.fonttype"]       = 42
    plt.rcParams["svg.fonttype"]      = "none"


def short(label: str) -> str:
    return SHORT_NAME.get(label, label)


def compact(label: str) -> str:
    return COMPACT_NAME.get(label, short(label))


# --------------------------------------------------------------------------- #
# Data discovery / loading
# --------------------------------------------------------------------------- #

def discover_files() -> list[tuple[str, str, bool]]:
    """Return [(label, path, is_original), ...] sorted by ORDER_PREFERENCE."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_with_distances.csv")))
    if not files:
        raise FileNotFoundError(
            f"No *_with_distances.csv found in {DATA_DIR!r}. "
            "Set CITATION_DATA_DIR to the directory holding the dyad-distance "
            "CSVs (distributed separately from this code).")
    out = []
    for fp in files:
        base = os.path.basename(fp).replace("_with_distances.csv", "")
        label = base.replace("llm_generated__", "").replace("original_dyads", "ORIGINAL")
        out.append((label, fp, base.startswith("original")))
    out.sort(key=lambda x: ORDER_PREFERENCE.index(x[0]) if x[0] in ORDER_PREFERENCE else 99)
    return out


def load_per_slot(file_labels) -> dict[str, pd.DataFrame]:
    """One DataFrame per file, one row per citation slot (all columns intact)."""
    out = {}
    for label, fp, _ in file_labels:
        out[label] = pd.read_csv(fp, low_memory=False)
        print(f"  {label:42s}  slot rows: {len(out[label]):>8,}")
    return out


def _bin_distances(d: np.ndarray) -> np.ndarray:
    out = np.full(len(d), "", dtype=object)
    out[d == -1] = "unreach"
    for k in range(5):
        out[d == k] = f"d={k}"
    out[d >= 5] = "d>=5"
    return out


def load_pooled(file_labels, llm_mot_col: str = "motivation_judge_filled"
                ) -> dict[str, pd.DataFrame]:
    """Pool the four dyad-distance columns into one long DataFrame per file.

    ORIGINAL groups by motivation_judge_original; LLMs by `llm_mot_col`
    (judge_filled by default, or motivation_self). Returned columns:
    focal_arxiv_id, context_index, position, dyad_col, d (int; -1=unreachable),
    mot (str), bin (str). Rows with a NaN distance are dropped.
    """
    data = {}
    for label, fp, is_orig in file_labels:
        mot_col = "motivation_judge_original" if is_orig else llm_mot_col
        usecols = DIST_COLS + [mot_col, "focal_arxiv_id", "context_index", "position"]
        df = pd.read_csv(fp, usecols=usecols, low_memory=False)
        pooled = pd.concat([
            pd.DataFrame({
                "focal_arxiv_id": df["focal_arxiv_id"],
                "context_index":  df["context_index"],
                "position":       df["position"],
                "d":              pd.to_numeric(df[c], errors="coerce"),
                "mot":            df[mot_col].fillna("<missing>"),
                "dyad_col":       c.replace("dist_", ""),
            })
            for c in DIST_COLS
        ], ignore_index=True)
        pooled = pooled.dropna(subset=["d"])
        pooled["d"]   = pooled["d"].astype(int)
        pooled["bin"] = _bin_distances(pooled["d"].values)
        data[label]   = pooled
        print(f"  {label:42s}  pooled rows: {len(pooled):>8,}")
    return data


def normalize_section(s: str) -> str:
    """Collapse near-duplicate section headings to the TOP_SECTIONS canon."""
    if not isinstance(s, str):
        return "<unknown>"
    t = s.strip().rstrip(".").strip()
    low = t.lower().replace("works", "work").replace("settings", "setup")
    CANON = {
        "introduction": "Introduction",
        "related work": "Related Work",
        "background": "Background",
        "method": "Method", "methods": "Method", "methodology": "Method",
        "approach": "Method",
        "experiments": "Experiments", "experiment": "Experiments",
        "experimental setup": "Experiments", "experiment setup": "Experiments",
        "setup": "Experiments", "baselines": "Experiments", "datasets": "Experiments",
        "implementation details": "Experiments",
        "results": "Results", "analysis": "Results",
        "discussion": "Discussion",
        "conclusion": "Conclusion", "conclusions": "Conclusion",
        "limitations": "Limitations",
    }
    return CANON.get(low, t[:30])


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

def sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"


def mean_ci95(arr: np.ndarray) -> tuple[float, float, int]:
    """(mean, 1.96*SE, n) for a 1-D array; NaNs for n < 2."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 2:
        return (float(a[0]) if n == 1 else float("nan"), float("nan"), n)
    return float(a.mean()), 1.96 * float(a.std(ddof=1) / np.sqrt(n)), n


def savefig(fig, stem: str, exts=("png",)) -> None:
    """Save `fig` as <FIG_DIR>/<stem>.<ext> for each requested extension."""
    for ext in exts:
        path = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)
