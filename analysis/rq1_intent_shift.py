"""
RQ1 - Intent shift from human citations.

"Do LLMs reproduce the distribution of citation intents observed in human
citations?" The answer is no: LLMs cite less critically, under-producing
contrasting citations and re-writing genuinely critical contexts as supporting.

Figures produced (PNG + PDF) into <FIG_DIR>:

  motivation_label_preservation.png
      For each LLM and each intent the judge assigned on the *human* sentence,
      how often the judge gives the SAME label on the LLM-filled sentence
      (the confusion-matrix diagonal). Coloured = preserved, grey = changed;
      bar height = #labeled slots; %-preserved annotated above each bar.
      Contrasting is the least preserved intent for every model.

  motivation_per_paper_aggregate.png
      Per-paper share of labeled slots in each intent, averaged across papers
      (one observation per focal paper) with 95% CIs. Dashed lines mark the
      human (ORIGINAL) per-intent level. Shows LLMs inflating supporting and
      suppressing contrasting relative to humans.

  step3_section_supporting_curve_normalized.png
  step3_section_contrasting_curve_normalized.png
      Per-section rate of supporting / contrasting citations, each curve divided
      by its own cross-section mean (1.0 = the file's own average), so the
      *shape* of section emphasis is comparable across models regardless of
      overall warmth.

Run:
    python rq1_intent_shift.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from common import (
    MOT_ORDER, MOT_COLORS, MODEL_BAND_COLORS, MODEL_BAND_ALPHA, TOP_SECTIONS,
    discover_files, load_per_slot, normalize_section, short, compact, savefig,
)


# --------------------------------------------------------------------------- #
# motivation_label_preservation
# --------------------------------------------------------------------------- #

def fig_label_preservation(slot_data, llm_labels):
    W = 0.26
    x = np.arange(len(llm_labels))
    fig, ax = plt.subplots(figsize=(5, 3))
    for xi, lbl in enumerate(llm_labels):
        ax.axvspan(xi - 0.5, xi + 0.5, color=MODEL_BAND_COLORS.get(lbl, "#808080"),
                   alpha=MODEL_BAND_ALPHA, linewidth=0, zorder=0)

    print("\n[motivation_label_preservation]  judge(original) -> judge(filled), per labeled slot")
    for i, m in enumerate(MOT_ORDER):
        cors, tots = [], []
        for lbl in llm_labels:
            df = slot_data[lbl]
            sub = df[df["motivation_judge_original"].isin(MOT_ORDER)
                     & df["motivation_judge_filled"].isin(MOT_ORDER)]
            rows_m = sub[sub["motivation_judge_original"] == m]
            tots.append(len(rows_m))
            cors.append(int((rows_m["motivation_judge_filled"] == m).sum()))
        cors, tots = np.array(cors), np.array(tots)
        inc = tots - cors
        xpos = x + (i - 1) * W
        ax.bar(xpos, cors, W, color=MOT_COLORS[m], alpha=0.7, edgecolor="white",
               lw=0.4, zorder=3)
        ax.bar(xpos, inc, W, bottom=cors, color="#cccccc", alpha=0.7,
               edgecolor="white", lw=0.4, zorder=3)
        for xp, tot, cor in zip(xpos, tots, cors):
            if tot:
                ax.text(xp, tot, f"{100 * cor / tot:.0f}%", ha="center", va="bottom",
                        fontsize=4.5, rotation=90, color="#333")
        for lbl, tot, cor in zip(llm_labels, tots, cors):
            print(f"   {short(lbl):18s} {m:11s} preserved={cor:>6,}/{tot:>6,} "
                  f"({100 * cor / max(tot, 1):5.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels([compact(l) for l in llm_labels], fontsize=7)
    ax.set_ylabel("# of citations", fontsize=8)
    ax.set_ylim(0, 95000)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title("Motivation preservation rate (LLMs vs Original)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3); ax.set_axisbelow(True)
    handles = [Patch(facecolor=MOT_COLORS[m], alpha=0.7, edgecolor="white",
                     label=f"{m} (preserved)") for m in MOT_ORDER]
    handles.append(Patch(facecolor="#cccccc", alpha=0.7, edgecolor="white", label="changed"))
    ax.legend(handles=handles, fontsize=4.6, frameon=False, ncol=2, loc="upper right",
              handlelength=1.2, columnspacing=1.0, handletextpad=0.4)
    plt.tight_layout()
    savefig(fig, "motivation_label_preservation", exts=("png", "pdf"))


# --------------------------------------------------------------------------- #
# motivation_per_paper_aggregate
# --------------------------------------------------------------------------- #

def _per_paper_pct(df, col, m):
    sub = df[df[col].isin(MOT_ORDER)]
    if len(sub) == 0:
        return np.array([], dtype=float)
    return (sub.groupby("focal_arxiv_id")[col]
            .apply(lambda s: 100.0 * (s == m).sum() / len(s)).values.astype(float))


def fig_per_paper_aggregate(slot_data, file_labels):
    stats = {}
    print("\n[motivation_per_paper_aggregate]  per-paper intent rate (mean +/- 95% CI across papers)")
    for lbl, _, is_orig in file_labels:
        col = "motivation_judge_original" if is_orig else "motivation_judge_filled"
        stats[lbl] = {}
        for m in MOT_ORDER:
            pp = _per_paper_pct(slot_data[lbl], col, m)
            if len(pp) >= 2:
                mu = float(np.mean(pp)); ci = 1.96 * float(np.std(pp, ddof=1) / np.sqrt(len(pp)))
            else:
                mu, ci = np.nan, np.nan
            stats[lbl][m] = (mu, ci, len(pp))

    labels = [l for l, _, _ in file_labels]
    xs = np.arange(len(labels)); W = 0.26
    fig, ax = plt.subplots(figsize=(5, 3))
    for xi, lbl in enumerate(labels):
        if lbl == "ORIGINAL":
            continue
        ax.axvspan(xi - 0.5, xi + 0.5, color=MODEL_BAND_COLORS.get(lbl, "#808080"),
                   alpha=MODEL_BAND_ALPHA, linewidth=0, zorder=0)
    for i, m in enumerate(MOT_ORDER):
        means = [stats[l][m][0] for l in labels]
        cis   = [stats[l][m][1] for l in labels]
        ax.bar(xs + (i - 1) * W, means, W, yerr=cis, color=MOT_COLORS[m], alpha=0.7,
               edgecolor="white", linewidth=0.4, zorder=3,
               error_kw=dict(ecolor="#444444", elinewidth=0.8, capsize=2, capthick=0.7))
        orig_mu = stats["ORIGINAL"][m][0]
        if np.isfinite(orig_mu):
            ax.axhline(orig_mu, color=MOT_COLORS[m], linestyle="--", alpha=0.6,
                       linewidth=1.0, zorder=2)
        for l in labels:
            mu, ci, npap = stats[l][m]
            print(f"   {short(l):18s} {m:11s} mean={mu:5.1f}%  95%CI=+/-{ci:4.1f}  n_papers={npap}")

    ax.set_xticks(xs)
    ax.set_xticklabels([compact(l) for l in labels], fontsize=5)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel("% of contexts", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title("Per-paper motivation rate (LLMs vs Original)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3); ax.set_axisbelow(True)
    handles = [Patch(facecolor=MOT_COLORS[m], alpha=0.7, edgecolor="white", label=m)
               for m in MOT_ORDER]
    handles.append(Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.0,
                          label="Original baseline"))
    ax.legend(handles=handles, fontsize=4.6, frameon=False, ncol=2, loc="upper right",
              handlelength=1.2, columnspacing=1.0, handletextpad=0.4)
    plt.tight_layout()
    savefig(fig, "motivation_per_paper_aggregate", exts=("png", "pdf"))


# --------------------------------------------------------------------------- #
# step3 section curves (self-normalized)
# --------------------------------------------------------------------------- #

def _section_long(slot_data, file_labels):
    rows = []
    for lbl, _, is_orig in file_labels:
        df = slot_data[lbl].copy()
        df["section_canon"] = df["section"].map(normalize_section)
        mot_col = "motivation_judge_original" if is_orig else "motivation_judge_filled"
        for sec in TOP_SECTIONS:
            sub = df.loc[df["section_canon"] == sec, mot_col]
            sub = sub[sub.isin(MOT_ORDER)]
            n = len(sub)
            counts = sub.value_counts() if n else pd.Series(dtype=int)
            for m in MOT_ORDER:
                pct = (100 * counts.get(m, 0) / n) if n else 0.0
                rows.append({"label": lbl, "section": sec, "mot": m, "pct": pct, "n": n})
    return pd.DataFrame(rows)


def fig_section_curves_normalized(df_long, file_labels):
    xs = np.arange(len(TOP_SECTIONS))
    for m, stem, title in [
        ("supporting",  "step3_section_supporting_curve_normalized",  "Supporting context"),
        ("contrasting", "step3_section_contrasting_curve_normalized", "Contrasting context"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 3))
        print(f"\n[{stem}]  {title} rate / file mean (1.0 = the file's own average)")
        plotted = []
        for lbl, _, is_orig in file_labels:
            ys = []
            for sec in TOP_SECTIONS:
                row = df_long[(df_long["label"] == lbl) & (df_long["section"] == sec)
                              & (df_long["mot"] == m)]
                ys.append(float(row["pct"].iloc[0]) if not row.empty else np.nan)
            mean = np.nanmean(ys) if np.any(~np.isnan(ys)) else np.nan
            ys = [(y / mean if mean and not np.isnan(mean) else np.nan) for y in ys]
            plotted.append(ys)
            color = "black" if is_orig else MODEL_BAND_COLORS.get(lbl, "#888888")
            if is_orig:
                ax.plot(xs, ys, linestyle="none", marker="o", markersize=5,
                        markerfacecolor=(1, 1, 1, 0.8), markeredgecolor="black",
                        markeredgewidth=0.7, label=compact(lbl), zorder=6)
            else:
                ax.plot(xs, ys, linestyle="none", marker="o", markersize=5, color=color,
                        alpha=0.8, markeredgecolor=color, markeredgewidth=0.7,
                        label=compact(lbl), zorder=4)
            print(f"   {short(lbl):18s} " +
                  "  ".join(f"{s[:4]}={y:.2f}" for s, y in zip(TOP_SECTIONS, ys)))

        ax.axhline(1.0, color="#888888", linestyle=":", linewidth=1.0, zorder=1)
        allv = np.array([v for row in plotted for v in row], dtype=float)
        ymax = (np.nanmax(allv) + 0.15) if np.any(~np.isnan(allv)) else 2.0
        ax.set_ylabel("Self-normalized rate", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(TOP_SECTIONS, fontsize=5)
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", length=0)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3); ax.set_axisbelow(True)
        # x-axis drawn as a left-to-right "paper writing order" arrow.
        ax.spines["bottom"].set_visible(False)
        ax.annotate("", xy=(1.035, 0.0), xytext=(-0.01, 0.0),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#333333", linewidth=1.3),
                    annotation_clip=False, zorder=2)
        ax.set_xlabel("Paper section orders", fontsize=7, color="#555555")
        ax.legend(fontsize=5.5, framealpha=0.6, ncol=2, loc="best")
        plt.tight_layout()
        savefig(fig, stem, exts=("png", "pdf"))


# --------------------------------------------------------------------------- #

def main():
    file_labels = discover_files()
    llm_labels = [l for l, _, is_orig in file_labels if not is_orig]
    print("Data points in scope:")
    print(f"  files            : {len(file_labels)} (ORIGINAL + {len(llm_labels)} LLMs)")
    slot_data = load_per_slot(file_labels)

    fig_label_preservation(slot_data, llm_labels)
    fig_per_paper_aggregate(slot_data, file_labels)
    df_long = _section_long(slot_data, file_labels)
    fig_section_curves_normalized(df_long, file_labels)
    print("\nRQ1 done.")


if __name__ == "__main__":
    from common import apply_style
    apply_style()
    main()
