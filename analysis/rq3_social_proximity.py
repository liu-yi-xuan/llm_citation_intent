"""
RQ3 - Citation and social proximity.

"Does coauthorship proximity shape LLM citation behavior across intents, as it
does for humans?" No. Humans cite within their close collaboration neighborhood
- especially for supporting citations - whereas every LLM draws on more socially
distant authors and shows no intent gradient.

For each citation slot we resolve focal & cited first/last authors to researcher
IDs and take BFS shortest-path distances on a 2015-2024 coauthorship network
(2.1M researchers, 20.3M edges) for four author-role dyads. d=0 is a self-cite,
d=1 a direct coauthor, larger = more distant; disconnected/missing = dropped.
Each citation is summarised by the mean reachable distance over its (up to four)
dyads; for the per-context analyses these are further averaged within a
(paper, context) slot. Intent uses judge_filled for LLMs, judge_original for
humans.

Figures produced into <FIG_DIR>:

  distance_kde_per_context_mean.png          (PNG + PDF + SVG)
      KDE of per-context mean reachable distance, one curve per file (intents
      pooled). Humans peak closer (~3.4) with a near-circle shoulder; LLMs sit
      to the right (3.65-3.89).

  errorbar_distance_combined_per_context__mean.png   (PNG + PDF + SVG)
      Mean per-context distance by intent, one panel, all files on a shared
      axis; per-model ANOVA + pairwise Welch t-test brackets. Humans show a
      drastic supporting < contrasting/mentioning gradient; LLMs cluster flat.

  near_network_d_le_1_combined.png           (PNG + PDF + SVG)
      Per-paper in-network citation rate (d <= 1: self-cite or direct coauthor)
      by intent. Humans reach 7-10%, elevated for supporting; every LLM stays
      below ~1.6% with no intent gradient.

Run:
    python rq3_social_proximity.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, f_oneway, ttest_ind

from common import (
    MOT_ORDER, MOT_COLORS, MODEL_BAND_COLORS, MODEL_BAND_ALPHA,
    discover_files, load_pooled, short, sig_stars, savefig, FIG_DIR,
)

FF_TAG = "focal_first_to_cited_first"


# --------------------------------------------------------------------------- #
# group builders
# --------------------------------------------------------------------------- #

def _near_network_per_paper_groups(pooled, distance):
    """Per-motivation list of per-PAPER rates (%) of dyads within `distance`
    hops. `distance` may be an int or an iterable of ints, e.g. (0, 1)."""
    if isinstance(distance, (int, np.integer)):
        targets = {f"d={int(distance)}"}
    else:
        targets = {f"d={int(d)}" for d in distance}
    groups = []
    for mot in MOT_ORDER:
        sub = pooled[pooled["mot"] == mot]
        if len(sub) == 0:
            groups.append(np.array([], dtype=float)); continue
        per_paper = (sub.groupby("focal_arxiv_id")["bin"]
                     .apply(lambda b: 100 * b.isin(targets).sum() / len(b))
                     .values.astype(float))
        groups.append(per_paper)
    return groups


def _mean_distance_groups(pooled, dyad_mode, agg_level):
    """Per-motivation list of reachable-distance observations (hops)."""
    groups = []
    for mot in MOT_ORDER:
        reach = pooled[(pooled["mot"] == mot) & (pooled["d"] >= 0)]
        if dyad_mode == "ff_only":
            per_row = reach[reach["dyad_col"] == FF_TAG]
        elif dyad_mode == "mean_of_4":
            per_row = reach.groupby(["focal_arxiv_id", "context_index", "position"],
                                    as_index=False)["d"].mean()
        elif dyad_mode == "all_pooled":
            per_row = reach[["focal_arxiv_id", "context_index", "position", "d"]]
        else:
            raise ValueError(f"unknown dyad_mode {dyad_mode!r}")
        if agg_level == "per_context":
            obs = per_row.groupby(["focal_arxiv_id", "context_index"])["d"].mean().values
        elif agg_level == "per_citation":
            obs = per_row["d"].values
        else:
            raise ValueError(f"unknown agg_level {agg_level!r}")
        groups.append(np.asarray(obs, dtype=float))
    return groups


# --------------------------------------------------------------------------- #
# shared one-axis errorbar figure (used by two of the three plots)
# --------------------------------------------------------------------------- #

def _fig_combined_one_axis(file_labels, groups_by_label, stem, title, ylabel,
                           legend_loc="lower right"):
    labels = [l for l, _, _ in file_labels]
    n = len(labels)
    offs = np.linspace(-0.26, 0.26, len(MOT_ORDER))
    fig, ax = plt.subplots(figsize=(6, 4))
    for xi, label in enumerate(labels):
        if label == "ORIGINAL":
            continue
        ax.axvspan(xi - 0.5, xi + 0.5, color=MODEL_BAND_COLORS.get(label, "#808080"),
                   alpha=MODEL_BAND_ALPHA, linewidth=0, zorder=0)

    print(f"\n[{stem}]  {title}  (mean +/- 95% CI; per-model ANOVA + pairwise Welch t)")
    tops = []
    sig_by_model = {}
    for xi, label in enumerate(labels):
        groups = groups_by_label[label]
        means, cis, ns = [], [], []
        for arr in groups:
            m = len(arr); ns.append(m)
            if m >= 2:
                means.append(float(np.mean(arr)))
                cis.append(1.96 * float(np.std(arr, ddof=1) / np.sqrt(m)))
            else:
                means.append(np.nan); cis.append(np.nan)
        valid = [g for g in groups if len(g) >= 5]
        try:
            aov = f_oneway(*valid)[1] if len(valid) >= 2 else float("nan")
        except ValueError:
            aov = float("nan")
        for off, mot, mu, ci in zip(offs, MOT_ORDER, means, cis):
            ax.errorbar(xi + off, mu, yerr=ci, fmt="o", markersize=6, color=MOT_COLORS[mot],
                        markerfacecolor=MOT_COLORS[mot], markeredgecolor="#222222",
                        markeredgewidth=0.7, ecolor="#444444", elinewidth=1.2,
                        capsize=4, capthick=1.0, zorder=5)
        finite_tops = [mu + (ci if np.isfinite(ci) else 0.0)
                       for mu, ci in zip(means, cis) if np.isfinite(mu)]
        tops.append((xi, max(finite_tops) if finite_tops else np.nan, aov))
        print(f"  {short(label)}:  ANOVA p={aov:.3g}")
        for mot, mu, ci, m in zip(MOT_ORDER, means, cis, ns):
            mu_s = f"{mu:.4f}" if np.isfinite(mu) else "nan"
            print(f"      {mot:11s} mean={mu_s}  95%CI=+/-{ci:.4f}  n={m}")
        drawn = []
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            g1, g2 = groups[i], groups[j]
            if len(g1) >= 5 and len(g2) >= 5:
                try:
                    p = ttest_ind(g1, g2, equal_var=False)[1]; st = sig_stars(p)
                except ValueError:
                    p = float("nan"); st = "ns"
            else:
                p = float("nan"); st = "n/a"
            p_s = f"{p:.3g}" if np.isfinite(p) else "nan"
            print(f"      {MOT_ORDER[i]} vs {MOT_ORDER[j]}: Welch p={p_s} {st}")
            if np.isfinite(p):
                drawn.append((i, j, st, "#222222") if p < 0.05 else (i, j, "ns", "#999999"))
        sig_by_model[xi] = drawn

    ax.set_xticks(range(n))
    ax.set_xticklabels([short(l) for l in labels], fontsize=5.7)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.6, n - 0.4)
    ax.grid(axis="y", linestyle="--", alpha=0.3); ax.set_axisbelow(True)

    ylo, yhi = ax.get_ylim(); yspan = yhi - ylo
    step = 0.06 * yspan; tick = 0.012 * yspan
    need_top = yhi
    for xi, top, _ in tops:
        pairs = sig_by_model.get(xi, [])
        if np.isfinite(top) and pairs:
            n_levels = max(j - i for (i, j, _l, _c) in pairs)
            need_top = max(need_top, top + (n_levels + 0.8) * step)
    ax.set_ylim(ylo, max(yhi, need_top))

    top_by_xi = {xi: top for xi, top, _ in tops}
    for xi, pairs in sig_by_model.items():
        top = top_by_xi.get(xi, np.nan)
        if not np.isfinite(top) or not pairs:
            continue
        base = top + 0.4 * step
        for (i, j, lab, color) in pairs:
            level = (j - i) - 1
            y = base + level * step
            x1, x2 = xi + offs[i], xi + offs[j]
            ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y], color=color,
                    linewidth=0.8, zorder=6)
            ax.text((x1 + x2) / 2, y + tick, lab, ha="center", va="bottom",
                    fontsize=6.5 if lab == "ns" else 8.5, color=color, zorder=6)

    handles = [Line2D([0], [0], marker="o", linestyle="", color=MOT_COLORS[m],
                      markeredgecolor="#222222", markersize=6, label=m) for m in MOT_ORDER]
    ax.legend(handles=handles, title="motivation", loc=legend_loc, fontsize=8.5,
              ncol=3, framealpha=0.6)
    plt.tight_layout()
    savefig(fig, stem, exts=("png", "pdf", "svg"))


# --------------------------------------------------------------------------- #
# distance_kde_per_context_mean
# --------------------------------------------------------------------------- #

def fig_mean_distance_kde(data, file_labels, bw=0.35):
    series, allv = [], []
    for label, _, is_orig in file_labels:
        reach = data[label][data[label]["d"] >= 0]
        if len(reach) == 0:
            continue
        per_row = reach.groupby(["focal_arxiv_id", "context_index", "position"])["d"].mean()
        per_ctx = per_row.groupby(level=[0, 1]).mean().values.astype(float)
        per_ctx = per_ctx[np.isfinite(per_ctx)]
        if len(per_ctx) < 5:
            continue
        series.append((label, is_orig, per_ctx)); allv.append(per_ctx)

    print("\n[distance_kde_per_context_mean]  KDE of per-context mean reachable distance")
    for label, _, per_ctx in series:
        print(f"  {short(label):18s} n_contexts={len(per_ctx):>6,}  "
              f"mean={per_ctx.mean():.3f}  median={np.median(per_ctx):.3f}")

    allv = np.concatenate(allv)
    lo, hi = np.percentile(allv, [0.5, 99.5])
    xs = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, is_orig, per_ctx in series:
        color = "black" if is_orig else MODEL_BAND_COLORS.get(label, "#888888")
        kde = gaussian_kde(per_ctx, bw_method=bw)
        ax.plot(xs, kde(xs), color=color, linewidth=1.6, alpha=1.0 if is_orig else 0.7,
                label=rf"{short(label)} ($\langle d\rangle$={per_ctx.mean():.2f})",
                zorder=6 if is_orig else 4)
        ax.axvline(per_ctx.mean(), color=color, linestyle="--", linewidth=1.0,
                   alpha=0.9 if is_orig else 0.6, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"$\langle d\rangle$", fontsize=10)
    ax.set_ylabel("density", fontsize=10)
    ax.set_title("Mean distance distributions", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3); ax.set_axisbelow(True)
    ax.legend(fontsize=6, framealpha=0.6, loc="upper left")
    plt.tight_layout()
    savefig(fig, "distance_kde_per_context_mean", exts=("png", "pdf", "svg"))


# --------------------------------------------------------------------------- #

def main():
    file_labels = discover_files()
    print("Data points in scope:")
    print(f"  files            : {len(file_labels)} (ORIGINAL + {len(file_labels) - 1} LLMs)")
    print(f"  figures dir       : {FIG_DIR}")
    data = load_pooled(file_labels, llm_mot_col="motivation_judge_filled")

    fig_mean_distance_kde(data, file_labels)

    _fig_combined_one_axis(
        file_labels,
        {lbl: _mean_distance_groups(data[lbl], "mean_of_4", "per_context")
         for lbl, _, _ in file_labels},
        stem="errorbar_distance_combined_per_context__mean",
        title="Mean distance distributions", ylabel=r"$\langle d\rangle$",
        legend_loc="lower right")

    _fig_combined_one_axis(
        file_labels,
        {lbl: _near_network_per_paper_groups(data[lbl], (0, 1))
         for lbl, _, _ in file_labels},
        stem="near_network_d_le_1_combined",
        title="In-network citation rate (d <= 1)", ylabel="probability (%)",
        legend_loc="upper right")
    print("\nRQ3 done.")


if __name__ == "__main__":
    from common import apply_style
    apply_style()
    main()
