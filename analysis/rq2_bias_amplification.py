"""
RQ2 - Citation bias amplified by rhetorical intent.

"Does citation intent modulate the divergence between LLM and human citations
in cited-paper attributes?" Yes: LLMs over-cite popular and older work, and the
human-LLM gap peaks at a *different* intent for each attribute - popularity at
supporting, recency at contrasting, team size at mentioning.

We compare three cited-paper attributes between each LLM and the human baseline,
split by the judge's intent label (judge_filled for LLMs, judge_original for
humans), on matched slots only (cited_dim_id present):

  recency   = focal_year - cited_year      (years; >0 => older work)
  popularity = cited_citations_count        (ratio LLM/human; >1 => more-cited)
  teamsize  = cited_teamsize                (ratio LLM/human; <1 => smaller teams)

Figures produced (PNG + PDF) into <FIG_DIR>:

  recency_gap_by_intent_perpaper.png    paper-level mean difference (LLM-human),
  popularity_gap_by_intent_perpaper.png paper-level geometric-mean ratio,
      both aggregated across focal papers so the 95% CI reflects between-paper
      variation rather than citation-level pseudo-replication. The most-deviant
      intent per model is outlined and starred (Welch t-test of its per-paper
      log-ratios/differences vs. a reference intent).

  teamsize_gap_by_intent.png            citation-level geometric-mean ratio gap,
      with pairwise difference-in-differences z-test stars on the most-deviant
      (smallest-ratio) intent.

Run:
    python rq2_bias_amplification.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from scipy.stats import ttest_ind, norm

from common import (
    MOT_ORDER, MOT_COLORS, discover_files, load_per_slot, short, compact,
    sig_stars, savefig,
)

EXTREME    = {"recency": "max", "citations": "max", "teamsize": "min"}
REF_INTENT = {"recency": "mentioning", "citations": "mentioning", "teamsize": "contrasting"}


# --------------------------------------------------------------------------- #
# citation-level gap (teamsize_gap_by_intent)
# --------------------------------------------------------------------------- #

def _intent_attr_means(slot_data, file_labels):
    """means[attr][label][mot] = (mean, 1.96*SE, n) over matched, in-range slots."""
    means = {"recency": {}, "citations": {}, "teamsize": {}}
    for label, _, is_orig in file_labels:
        df = slot_data[label]
        mot_col = "motivation_judge_original" if is_orig else "motivation_judge_filled"
        sub = df.dropna(subset=["cited_dim_id"])
        sub = sub[sub[mot_col].isin(MOT_ORDER)]
        rec  = (sub["focal_year"] - sub["cited_year"]).astype(float)
        cit  = sub["cited_citations_count"].astype(float)
        team = sub["cited_teamsize"].astype(float)
        for attr in means:
            means[attr][label] = {}
        for m in MOT_ORDER:
            sel = (sub[mot_col] == m).values
            rv = rec[sel].values;  rv = rv[(rv >= 0) & (rv <= 30)]
            cv = cit[sel].values;  cv = cv[cv > 0]
            tv = team[sel].values; tv = tv[tv > 0]
            for attr, v in (("recency", rv), ("citations", cv), ("teamsize", tv)):
                if len(v) >= 2:
                    mu = float(np.mean(v)); ci = 1.96 * float(np.std(v, ddof=1) / np.sqrt(len(v)))
                elif len(v) == 1:
                    mu, ci = float(v[0]), float("nan")
                else:
                    mu, ci = float("nan"), float("nan")
                means[attr][label][m] = (mu, ci, int(len(v)))
    return means


def fig_teamsize_gap_citation_level(slot_data, file_labels):
    means = _intent_attr_means(slot_data, file_labels)
    orig = next(l for l, _, io in file_labels if io)
    llms = [l for l, _, io in file_labels if not io]
    attr, ref, ylab, title = "teamsize", 1.0, "team-size gap: LLM/Human", "Team-size gap by intent"

    # ratio gap LLM/human and its propagated CI per (llm, motivation)
    gap, ci = {}, {}
    for l in llms:
        gap[l], ci[l] = {}, {}
        for m in MOT_ORDER:
            ml, cl, _ = means[attr][l][m]
            mh, ch, _ = means[attr][orig][m]
            if np.isfinite(ml) and np.isfinite(mh) and mh and ml:
                r = ml / mh
                gap[l][m] = r
                ci[l][m] = float(r * np.hypot(cl / ml, ch / mh))
            else:
                gap[l][m] = ci[l][m] = float("nan")

    def _pair_p(l, i, j):
        gi, gj = gap[l][MOT_ORDER[i]], gap[l][MOT_ORDER[j]]
        ei, ej = ci[l][MOT_ORDER[i]], ci[l][MOT_ORDER[j]]
        if not all(np.isfinite(v) for v in (gi, gj, ei, ej)):
            return float("nan")
        se = float(np.hypot(ei / 1.96, ej / 1.96))
        return float(2 * norm.sf(abs(gi - gj) / se)) if se else float("nan")

    print(f"\n[teamsize_gap_by_intent]  LLM/human ratio per intent (citation-level)")
    for m in MOT_ORDER:
        print(f"   human {m:11s}: teamsize mean={means[attr][orig][m][0]:.2f}  "
              f"n={means[attr][orig][m][2]:,}")
    for l in llms:
        print(f"   {short(l):18s} " +
              "  ".join(f"{m}={gap[l][m]:.2f}+/-{ci[l][m]:.2f}" for m in MOT_ORDER))

    ex = {}
    for l in llms:
        vals = {m: gap[l][m] for m in MOT_ORDER if np.isfinite(gap[l][m])}
        ex[l] = (min(vals, key=vals.get) if vals else None)   # smallest ratio

    x = np.arange(len(llms)); W = 0.26
    fig, ax = plt.subplots(figsize=(4, 3))
    for i, m in enumerate(MOT_ORDER):
        edge = [MOT_COLORS[m] if ex[l] == m else "none" for l in llms]
        lw   = [1.4 if ex[l] == m else 0.0 for l in llms]
        ax.bar(x + (i - 1) * W, [gap[l][m] for l in llms], width=W,
               color=to_rgba(MOT_COLORS[m], 0.6), edgecolor=edge, linewidth=lw,
               yerr=[ci[l][m] for l in llms],
               error_kw=dict(ecolor="#333333", elinewidth=0.8, capsize=1.5), zorder=3)
    ax.axhline(ref, color="#666", linewidth=1.0, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([compact(l) for l in llms], fontsize=6)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3); ax.set_axisbelow(True)
    ylo, yhi = ax.get_ylim(); ax.set_ylim(ylo, yhi + 0.20 * (yhi - ylo))
    ylo, yhi = ax.get_ylim(); yspan = yhi - ylo
    ref_intent = REF_INTENT[attr]
    for jx, l in enumerate(llms):
        e_m = ex[l]
        if e_m is None or e_m == ref_intent or not np.isfinite(gap[l].get(ref_intent, np.nan)):
            continue
        p = _pair_p(l, MOT_ORDER.index(e_m), MOT_ORDER.index(ref_intent))
        e = ci[l][e_m] if np.isfinite(ci[l][e_m]) else 0.0
        ax.text(x[jx] + (MOT_ORDER.index(e_m) - 1) * W, gap[l][e_m] + e + 0.02 * yspan,
                sig_stars(p), ha="center", va="bottom", fontsize=7, color="#333", zorder=6)
    ax.legend(handles=[Patch(facecolor=to_rgba(MOT_COLORS[m], 0.6), label=m) for m in MOT_ORDER],
              title="motivation", fontsize=6, title_fontsize=6.5, framealpha=0.6,
              loc="upper left", handlelength=1.2, borderpad=0.3, labelspacing=0.25)
    plt.tight_layout()
    savefig(fig, "teamsize_gap_by_intent", exts=("png", "pdf"))


# --------------------------------------------------------------------------- #
# paper-level gaps (recency / popularity)
# --------------------------------------------------------------------------- #

def fig_gap_perpaper(slot_data, file_labels):
    orig = next(l for l, _, io in file_labels if io)
    llms = [l for l, _, io in file_labels if not io]

    def _pp_means(df, mot_col, attr):
        sub = df.dropna(subset=["cited_dim_id"])
        sub = sub[sub[mot_col].isin(MOT_ORDER)].copy()
        if attr == "recency":
            sub["_v"] = (sub["focal_year"] - sub["cited_year"]).astype(float)
            sub = sub[(sub["_v"] >= 0) & (sub["_v"] <= 30)]
        elif attr == "citations":
            sub["_v"] = sub["cited_citations_count"].astype(float); sub = sub[sub["_v"] > 0]
        else:
            sub["_v"] = sub["cited_teamsize"].astype(float); sub = sub[sub["_v"] > 0]
        return sub.groupby(["focal_arxiv_id", mot_col])["_v"].mean()

    def _pp_arrays(llm, attr, kind):
        hm = _pp_means(slot_data[orig], "motivation_judge_original", attr)
        lm = _pp_means(slot_data[llm], "motivation_judge_filled", attr)
        h_lvl = set(hm.index.get_level_values(1)); l_lvl = set(lm.index.get_level_values(1))
        out = {}
        for m in MOT_ORDER:
            h = hm.xs(m, level=1) if m in h_lvl else pd.Series(dtype=float)
            l = lm.xs(m, level=1) if m in l_lvl else pd.Series(dtype=float)
            j = pd.concat([h.rename("h"), l.rename("l")], axis=1, join="inner").dropna()
            if kind == "diff":
                out[m] = (j["l"] - j["h"]).to_numpy()
            else:
                j = j[(j["h"] > 0) & (j["l"] > 0)]
                out[m] = (np.log(j["l"]) - np.log(j["h"])).to_numpy()
        return out

    def _agg(a, kind):
        n = len(a)
        if n < 2:
            return (float("nan"),) * 3 + (n,)
        mu = float(np.mean(a)); se = float(np.std(a, ddof=1) / np.sqrt(n))
        if kind == "diff":
            return mu, mu - 1.96 * se, mu + 1.96 * se, n
        return float(np.exp(mu)), float(np.exp(mu - 1.96 * se)), float(np.exp(mu + 1.96 * se)), n

    specs = [
        ("recency",   "diff",  0.0, "recency gap: LLM-Human (yr)",
         "Recency gap by intent", "recency_gap_by_intent_perpaper"),
        ("citations", "ratio", 1.0, "citation gap: LLM/Human",
         "Citation gap by intent", "popularity_gap_by_intent_perpaper"),
    ]
    x = np.arange(len(llms)); W = 0.26
    for attr, kind, ref, ylab, title, stem in specs:
        arrays = {l: _pp_arrays(l, attr, kind) for l in llms}
        pt = {l: {m: _agg(arrays[l][m], kind) for m in MOT_ORDER} for l in llms}
        print(f"\n[{stem}]  ({'difference' if kind == 'diff' else 'geometric-mean ratio'})")
        for l in llms:
            print(f"   {short(l):18s} " +
                  "  ".join(f"{m}={pt[l][m][0]:.2f}[{pt[l][m][1]:.2f},{pt[l][m][2]:.2f}]n={pt[l][m][3]}"
                            for m in MOT_ORDER))

        direction = EXTREME[attr]
        ex = {l: (max if direction == "max" else min)(
                 (m for m in MOT_ORDER if np.isfinite(pt[l][m][0])),
                 key=lambda m: pt[l][m][0], default=None) for l in llms}
        fig, ax = plt.subplots(figsize=(4, 3))
        for i, m in enumerate(MOT_ORDER):
            heights = [pt[l][m][0] for l in llms]
            lo = [pt[l][m][0] - pt[l][m][1] for l in llms]
            hi = [pt[l][m][2] - pt[l][m][0] for l in llms]
            edge = [MOT_COLORS[m] if ex[l] == m else "none" for l in llms]
            lw   = [1.4 if ex[l] == m else 0.0 for l in llms]
            ax.bar(x + (i - 1) * W, heights, W, color=to_rgba(MOT_COLORS[m], 0.6),
                   edgecolor=edge, linewidth=lw, yerr=np.array([lo, hi]),
                   error_kw=dict(ecolor="#333333", elinewidth=0.8, capsize=1.5), zorder=3)
        ax.axhline(ref, color="#666", lw=1.0, linestyle="--", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([compact(l) for l in llms], fontsize=5.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3); ax.set_axisbelow(True)
        ylo, yhi = ax.get_ylim(); ax.set_ylim(ylo, yhi + 0.18 * (yhi - ylo))
        ylo, yhi = ax.get_ylim(); yspan = yhi - ylo
        for jx, l in enumerate(llms):
            e_m = ex[l]
            if e_m is None:
                continue
            a1 = arrays[l][e_m]
            rest = [arrays[l][m] for m in MOT_ORDER if m != e_m and len(arrays[l][m])]
            a2 = np.concatenate(rest) if rest else np.array([])
            p = (ttest_ind(a1, a2, equal_var=False)[1]
                 if len(a1) >= 5 and len(a2) >= 5 else float("nan"))
            top = pt[l][e_m][2] if np.isfinite(pt[l][e_m][2]) else pt[l][e_m][0]
            ax.text(x[jx] + (MOT_ORDER.index(e_m) - 1) * W, top + 0.02 * yspan,
                    sig_stars(p), ha="center", va="bottom", fontsize=6.5, color="#333", zorder=6)
        ax.legend(handles=[Patch(facecolor=to_rgba(MOT_COLORS[m], 0.6), label=m) for m in MOT_ORDER],
                  title="motivation", fontsize=6, title_fontsize=6.5, framealpha=0.6,
                  loc="upper right", handlelength=1.2, borderpad=0.3, labelspacing=0.25)
        plt.tight_layout()
        savefig(fig, stem, exts=("png", "pdf"))


# --------------------------------------------------------------------------- #

def main():
    file_labels = discover_files()
    print("Data points in scope:")
    print(f"  files            : {len(file_labels)} (ORIGINAL + {len(file_labels) - 1} LLMs)")
    print(f"  attributes        : recency, popularity (citations), teamsize")
    slot_data = load_per_slot(file_labels)

    fig_gap_perpaper(slot_data, file_labels)         # recency + popularity (per-paper)
    fig_teamsize_gap_citation_level(slot_data, file_labels)  # teamsize (citation-level)
    print("\nRQ2 done.")


if __name__ == "__main__":
    from common import apply_style
    apply_style()
    main()
