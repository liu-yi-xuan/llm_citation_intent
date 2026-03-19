#%%
"""
Script 04: Criticism Signal Term Analysis
==========================================
Compare the frequency of criticism/contrast signal terms between:
  - Original citation sentences (ground truth)
  - LLM-generated citation sentences (GPT-4o responses)

Based on established disagreement signal terms from the literature
(challenge, conflict, contradict, contrary, contrast, controversy,
debate, differ, disagree, disprove, no consensus, questionable, refute).

If the LLM systematically avoids these terms compared to the originals,
that's evidence of citation sycophancy.

Usage:
    python 04_criticism_analysis.py
"""

import os
import re
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

RESPONSES_DIR = "data/arxiv_llm_responses"

# ============================================================
# Signal terms (from the literature on disagreement in science)
# Using regex patterns to capture variants (e.g., challenge*)
# ============================================================

SIGNAL_TERMS = {
    "challenge*":     r'\bchalleng\w*\b',
    "conflict*":      r'\bconflict\w*\b',
    "contradict*":    r'\bcontradict\w*\b',
    "contrary":       r'\bcontrary\b',
    "contrast*":      r'\bcontrast\w*\b',
    "controvers*":    r'\bcontrovers\w*\b',
    "debat*":         r'\bdebat\w*\b',
    "differ*":        r'\bdiffer(?!ent\w*)\w*\b',  # differ* but not different*
    "disagree*":      r'\bdisagree\w*\b',
    "disprov*":       r'\bdisprov\w*\b',
    "no consensus":   r'\bno\s+consensus\b|lack\s+of\s+consensus\b',
    "questionable":   r'\bquestionable\b',
    "refut*":         r'\brefut(?!ab)\w*\b',        # refut* but not refutab*
    # Additional terms common in NLP papers
    "outperform*":    r'\boutperform\w*\b',
    "underperform*":  r'\bunderperform\w*\b',
    "fail*":          r'\bfail\w*\b',
    "limit*":         r'\blimit\w*\b',
    "shortcoming*":   r'\bshortcoming\w*\b',
    "drawback*":      r'\bdrawback\w*\b',
    "inferior":       r'\binferior\b',
    "worse":          r'\bworse\b',
    "despite":        r'\bdespite\b',
    "however":        r'\bhowever\b',
    "although":       r'\balthough\b',
    "unlike":         r'\bunlike\b',
}


def count_signal_terms(text: str) -> dict:
    """Count occurrences of each signal term in a text string."""
    if not text or not isinstance(text, str):
        return {term: 0 for term in SIGNAL_TERMS}

    text_lower = text.lower()
    counts = {}
    for term, pattern in SIGNAL_TERMS.items():
        counts[term] = len(re.findall(pattern, text_lower))
    return counts


# ============================================================
# Load all LLM responses
# ============================================================

print("Loading LLM responses...")
response_files = glob.glob(os.path.join(RESPONSES_DIR, "*/responses.json"))
print(f"  Found {len(response_files)} papers with responses")

records = []
for rf in response_files:
    with open(rf) as f:
        data = json.load(f)

    for r in data.get("results", []):
        llm = r.get("llm_response", {})
        if "_error" in llm:
            continue

        records.append({
            "arxiv_id": data.get("arxiv_id"),
            "year": data.get("year"),
            "section": r.get("section", ""),
            "original_sentence": r.get("citation_sentence_original", ""),
            "llm_sentence": llm.get("citation_sentence", ""),
            "llm_motivation": llm.get("motivation", ""),
            "llm_confidence": llm.get("confidence", ""),
        })

df = pd.DataFrame(records)
print(f"  Total citation contexts: {len(df)}")

if len(df) == 0:
    print("No data to analyze. Run 03_llm_cite_recommend.py first.")
    exit()

# ============================================================
# Count signal terms in original vs LLM sentences
# ============================================================

print("Counting signal terms...")

# Count for originals
orig_counts = df["original_sentence"].apply(count_signal_terms)
orig_df = pd.DataFrame(orig_counts.tolist())

# Count for LLM
llm_counts = df["llm_sentence"].apply(count_signal_terms)
llm_df = pd.DataFrame(llm_counts.tolist())

# Aggregate: total occurrences per term
orig_totals = orig_df.sum().sort_values(ascending=False)
llm_totals = llm_df.sum().reindex(orig_totals.index)

# Also compute: fraction of sentences containing each term (at least once)
orig_presence = (orig_df > 0).mean()
llm_presence = (llm_df > 0).mean()

# ============================================================
# Print summary table
# ============================================================

print(f"\n{'=' * 75}")
print("SIGNAL TERM FREQUENCY COMPARISON")
print(f"{'=' * 75}")
print(f"  {'Term':<20s} {'Original':>10s} {'LLM':>10s} {'Ratio':>10s} {'Orig %':>10s} {'LLM %':>10s}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for term in orig_totals.index:
    o = orig_totals[term]
    l = llm_totals[term]
    ratio = l / o if o > 0 else float('inf')
    op = orig_presence[term] * 100
    lp = llm_presence[term] * 100
    print(f"  {term:<20s} {o:>10.0f} {l:>10.0f} {ratio:>10.2f} {op:>9.1f}% {lp:>9.1f}%")

total_orig = orig_totals.sum()
total_llm = llm_totals.sum()
print(f"  {'TOTAL':<20s} {total_orig:>10.0f} {total_llm:>10.0f} {total_llm/max(total_orig,1):>10.2f}")

# ============================================================
# LLM motivation distribution
# ============================================================

print(f"\n{'=' * 75}")
print("LLM MOTIVATION DISTRIBUTION")
print(f"{'=' * 75}")
mot_counts = df["llm_motivation"].value_counts()
for mot, cnt in mot_counts.items():
    print(f"  {mot:<20s}: {cnt:>6d} ({100*cnt/len(df):.1f}%)")

# ============================================================
# Plot: Horizontal bar chart — raw counts, Original vs LLM
# ============================================================

# Filter to terms that appear at least once in either source
mask = (orig_totals > 0) | (llm_totals > 0)
plot_terms = orig_totals[mask].index.tolist()

# Sort alphabetically
plot_terms = sorted(plot_terms)

# PiYG colors: green end for Original, pink end for LLM
cmap = plt.cm.PiYG
color_orig = cmap(0.8)   # green side
color_llm = cmap(0.2)    # pink side

# --- Compute per-paper signal term counts ---
df["orig_total_signals"] = orig_df[plot_terms].sum(axis=1)
df["llm_total_signals"] = llm_df[plot_terms].sum(axis=1)

paper_orig = df.groupby("arxiv_id")["orig_total_signals"].agg(["mean", "std"])
paper_llm = df.groupby("arxiv_id")["llm_total_signals"].agg(["mean", "std"])

fig, axes = plt.subplots(1, 2, figsize=(10, 7), dpi=300,
                          gridspec_kw={'width_ratios': [1, 1]})

# --- Left: Horizontal bar chart (total counts) ---
ax = axes[0]
y_pos = np.arange(len(plot_terms))
bar_height = 0.35

ax.barh(y_pos + bar_height/2, [orig_totals[t] for t in plot_terms],
        height=bar_height, color=color_orig, label='Original', alpha=0.9)
ax.barh(y_pos - bar_height/2, [llm_totals[t] for t in plot_terms],
        height=bar_height, color=color_llm, label='LLM (GPT-4o)', alpha=0.9)

ax.set_yticks(y_pos)
ax.set_yticklabels(plot_terms, fontsize=9)
ax.set_xlabel('Total Occurrences', fontsize=10)
ax.set_title('Criticism Terms (Total Counts)', fontsize=11)
ax.legend(fontsize=9, loc='lower right')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Right: Per-paper pointplot by term (horizontal, mean ± std) ---
ax = axes[1]

# Compute per-paper mean for each term, then get mean ± std across papers
for i, term in enumerate(plot_terms):
    # Per-paper mean count for this term
    df["_orig_term"] = orig_df[term]
    df["_llm_term"] = llm_df[term]

    paper_orig_term = df.groupby("arxiv_id")["_orig_term"].mean()
    paper_llm_term = df.groupby("arxiv_id")["_llm_term"].mean()

    # Original: point + errorbar
    ax.errorbar(paper_orig_term.mean(), i + bar_height/2,
                xerr=paper_orig_term.std(), fmt='o', markersize=5,
                capsize=3, capthick=1.2, linewidth=1.2,
                color=color_orig, ecolor=color_orig, alpha=0.9, zorder=3)

    # LLM: point + errorbar
    ax.errorbar(paper_llm_term.mean(), i - bar_height/2,
                xerr=paper_llm_term.std(), fmt='o', markersize=5,
                capsize=3, capthick=1.2, linewidth=1.2,
                color=color_llm, ecolor=color_llm, alpha=0.9, zorder=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(plot_terms, fontsize=9)
ax.set_xlabel('Avg per Paper (mean ± std)', fontsize=10)
ax.set_title('Per-Paper Term Rate', fontsize=11)
ax.legend(
    handles=[
        plt.Line2D([0], [0], marker='o', color=color_orig, linestyle='-',
                   markersize=5, label='Original'),
        plt.Line2D([0], [0], marker='o', color=color_llm, linestyle='-',
                   markersize=5, label='LLM (GPT-4o)'),
    ],
    fontsize=8, loc='lower right'
)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Clean up temp columns
df.drop(columns=["_orig_term", "_llm_term"], inplace=True, errors='ignore')

plt.tight_layout()
plt.savefig('data/criticism_analysis.png', bbox_inches='tight', dpi=300)
plt.show()
print(f"\nPlot saved to: data/criticism_analysis.png")

# ============================================================
# Plot 2: Motivation distribution
# ============================================================

fig, ax = plt.subplots(figsize=(5, 3), dpi=300)

mot_order = ['supporting', 'contrasting', 'mentioning']
mot_colors = {
    'supporting': 'salmon',
    'contrasting': 'skyblue',
    'mentioning': 'lightgreen',
}

counts = [mot_counts.get(m, 0) for m in mot_order]
colors = [mot_colors.get(m, '#999999') for m in mot_order]

ax.barh(mot_order, counts, color=colors, height=0.5, alpha=0.9)
for i, (m, c) in enumerate(zip(mot_order, counts)):
    pct = 100 * c / len(df)
    ax.text(c + max(counts)*0.02, i, f'{c:,} ({pct:.1f}%)', va='center', fontsize=9)

ax.set_xlabel('Count', fontsize=10)
ax.set_title('LLM Citation Motivation Distribution', fontsize=11)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('data/motivation_distribution.png', bbox_inches='tight', dpi=300)
plt.show()
print(f"Plot saved to: data/motivation_distribution.png")


# %%
RESPONSES_DIR = "data/arxiv_llm_responses"
RESPONSES_03B_DIR = "data/arxiv_original_llm_responses"

# ============================================================
# Load 03b outputs for motivation-vs-judge comparison
# ============================================================

print("\nLoading 03b judge responses...")
judge_files = glob.glob(os.path.join(RESPONSES_03B_DIR, "*/responses.json"))
print(f"  Found {len(judge_files)} papers with 03b responses")

def _norm_mot(v):
    if not isinstance(v, str):
        return ""
    v = v.strip().lower()
    return v if v in {"supporting", "contrasting", "mentioning"} else ""

judge_records = []
for jf in judge_files:
    with open(jf) as f:
        data_j = json.load(f)

    for r in data_j.get("results", []):
        m_self = _norm_mot(r.get("motivation_script03_self", {}).get("motivation", ""))
        m_judge_orig = _norm_mot(r.get("motivation_llm_as_judge_original", {}).get("motivation", ""))
        m_judge_fill = _norm_mot(r.get("motivation_llm_as_judge_filled", {}).get("motivation", ""))

        # keep rows where all three are valid so comparison uses the same set
        if not (m_self and m_judge_orig and m_judge_fill):
            continue

        judge_records.append({
            "arxiv_id": data_j.get("arxiv_id"),
            "context_index": r.get("context_index"),
            "llm_self_motivation": m_self,
            "judge_original_motivation": m_judge_orig,
            "judge_filled_motivation": m_judge_fill,
        })

df_judge = pd.DataFrame(judge_records)
print(f"  Comparable contexts (same-set): {len(df_judge)}")
# %%
# ============================================================
# Plot 2: Motivation distribution comparison (same-set, from 03b)
# ============================================================

if len(df_judge) == 0:
    print("No comparable 03b motivation data found for plotting.")
else:
    mot_order = ['supporting', 'contrasting', 'mentioning']
    mot_colors = {
        'supporting': 'salmon',
        'contrasting': 'skyblue',
        'mentioning': 'lightgreen',
    }

    c_self = df_judge["llm_self_motivation"].value_counts()
    c_orig = df_judge["judge_original_motivation"].value_counts()
    c_fill = df_judge["judge_filled_motivation"].value_counts()

    n = len(df_judge)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=300, sharey=True)

    panels = [
        ("LLM Self (Step 03)", c_self),
        ("Judge on Original", c_orig),
        ("Judge on Generated", c_fill),
    ]

    max_count = max(
        max([c_self.get(m, 0) for m in mot_order] + [1]),
        max([c_orig.get(m, 0) for m in mot_order] + [1]),
        max([c_fill.get(m, 0) for m in mot_order] + [1]),
    )

    for ax, (title, counts_map) in zip(axes, panels):
        counts = [counts_map.get(m, 0) for m in mot_order]
        colors = [mot_colors[m] for m in mot_order]

        ax.barh(mot_order, counts, color=colors, height=0.5, alpha=0.9)
        for i, (m, c) in enumerate(zip(mot_order, counts)):
            pct = 100 * c / n if n else 0
            ax.text(c + max_count * 0.02, i, f"{c:,} ({pct:.1f}%)", va="center", fontsize=8)

        ax.set_xlim(0, max_count * 1.25)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Count", fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel("Motivation", fontsize=9)
    fig.suptitle("Citation Contexts Distribution", fontsize=11)

    plt.tight_layout()
    plt.savefig("data/motivation_distribution_compare_03_vs_03b.png", bbox_inches="tight", dpi=300)
    plt.show()
    print("Plot saved to: data/motivation_distribution_compare_03_vs_03b.png")
# %%
