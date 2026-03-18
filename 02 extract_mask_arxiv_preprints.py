#%%
"""
Script 2: Extract & Parse Citation Contexts with Level-2 Masking
================================================================
Saves results PER PAPER under:
    data/arxiv_masked/{arxiv_id}/contexts.json
    data/arxiv_masked/{arxiv_id}/bib_entries.json
    data/arxiv_masked/{arxiv_id}/metadata.json

This way:
  - Each paper's results are independent
  - You can resume after interruption (skips already-processed papers)
  - Easy to inspect individual papers
  - No risk of losing everything on a crash

Usage:
    python 02_extract_and_mask.py
"""

import os
import re
import json
import gzip
import tarfile
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Timeout per paper (seconds) — skip papers that hang
TIMEOUT_SECONDS = 60


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "data/preprints_with_source_paths.csv"
EXTRACT_DIR = "data/arxiv_extracted"
OUTPUT_DIR = "data/arxiv_masked"


# ============================================================
# STEP 1: Extract LaTeX from tar.gz
# ============================================================

def clean_arxiv_id(arxiv_id: str) -> str:
    if pd.isna(arxiv_id):
        return None
    return arxiv_id.replace("arXiv:", "").replace("arxiv:", "").strip()


def extract_source(source_path: str, arxiv_id: str) -> str:
    if not source_path or not os.path.exists(source_path):
        return None

    clean_id = clean_arxiv_id(arxiv_id).replace("/", "_")
    extract_dir = os.path.join(EXTRACT_DIR, clean_id)

    if os.path.exists(extract_dir):
        for root, dirs, files in os.walk(extract_dir):
            if any(f.endswith(".tex") for f in files):
                return extract_dir

    os.makedirs(extract_dir, exist_ok=True)

    try:
        with tarfile.open(source_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        return extract_dir
    except tarfile.TarError:
        pass

    try:
        with gzip.open(source_path, "rb") as gz:
            content = gz.read()
        if b"\\begin{document}" in content or b"\\documentclass" in content:
            with open(os.path.join(extract_dir, "main.tex"), "wb") as f:
                f.write(content)
            return extract_dir
    except Exception:
        pass

    return None


def find_main_tex(extract_dir: str) -> str:
    if not extract_dir:
        return None

    tex_files = []
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.endswith(".tex"):
                tex_files.append(os.path.join(root, f))

    if not tex_files:
        return None

    for tex_file in tex_files:
        try:
            with open(tex_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if "\\begin{document}" in content:
                return tex_file
        except Exception:
            continue

    return max(tex_files, key=os.path.getsize)


# ============================================================
# STEP 2: Parse .bib files
# ============================================================

def extract_bib_entries(extract_dir: str) -> dict:
    if not extract_dir:
        return {}

    entries = {}
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if not f.endswith(".bib"):
                continue
            bib_path = os.path.join(root, f)
            try:
                with open(bib_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()

                for match in re.finditer(
                    r'@\w+\{([^,]+),\s*(.*?)\n\}', content, re.DOTALL
                ):
                    key = match.group(1).strip()
                    body = match.group(2)
                    entry = {}
                    for field in ["title", "author", "year", "journal",
                                  "booktitle", "url", "doi"]:
                        fm = re.search(
                            rf'{field}\s*=\s*[\{{"](.*?)[\}}"]\s*[,\n]',
                            body, re.IGNORECASE | re.DOTALL
                        )
                        if fm:
                            entry[field] = fm.group(1).strip()
                    if entry:
                        entries[key] = entry
            except Exception:
                continue

    return entries


# ============================================================
# STEP 3: Paragraph-level citation extraction with masking
# ============================================================

def clean_tex_content(content: str) -> str:
    lines = content.split("\n")
    cleaned = []
    for line in lines:
        cleaned.append(re.sub(r'(?<!\\)%.*$', '', line))
    return "\n".join(cleaned)


def get_section_at_position(pos: int, section_positions: list) -> str:
    current = "Preamble"
    for sec_pos, sec_name in section_positions:
        if sec_pos <= pos:
            current = sec_name
        else:
            break
    return current


def split_into_paragraphs(content: str) -> list:
    paragraphs = []
    para_pattern = re.compile(r'\n\s*\n')

    last_end = 0
    for match in para_pattern.finditer(content):
        para_text = content[last_end:match.start()].strip()
        if para_text:
            paragraphs.append((last_end, match.start(), para_text))
        last_end = match.end()

    remaining = content[last_end:].strip()
    if remaining:
        paragraphs.append((last_end, len(content), remaining))

    return paragraphs


def find_citation_sentence(paragraph: str, cite_match_start: int,
                           cite_match_end: int, para_start: int) -> dict:
    rel_start = cite_match_start - para_start

    # Protect abbreviations before splitting
    protected = paragraph
    abbreviations = ['et al.', 'Fig.', 'Eq.', 'Sec.', 'Tab.', 'Ref.',
                     'Exp.', 'No.', 'vs.', 'Dr.', 'Mr.', 'Mrs.', 'Prof.',
                     'i.e.', 'e.g.', 'etc.', 'approx.', 'resp.']
    for abbr in abbreviations:
        protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))

    sentence_pattern = re.compile(r'[.!?]\s+(?=[A-Z\\])')

    boundaries = [0]
    for m in sentence_pattern.finditer(protected):
        boundaries.append(m.end())
    boundaries.append(len(paragraph))

    cite_sentence_idx = None
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= rel_start < boundaries[i + 1]:
            cite_sentence_idx = i
            break

    if cite_sentence_idx is None:
        return {
            "sentence": paragraph,
            "sentence_start": 0,
            "sentence_end": len(paragraph),
            "before_sentence": "",
            "after_sentence": "",
        }

    s_start = boundaries[cite_sentence_idx]
    s_end = boundaries[cite_sentence_idx + 1]

    # One sentence BEFORE the citation sentence (immediate predecessor)
    if cite_sentence_idx > 0:
        prev_start = boundaries[cite_sentence_idx - 1]
        before_sentence = paragraph[prev_start:s_start].strip()
    else:
        before_sentence = ""

    # One sentence AFTER the citation sentence (immediate successor)
    if cite_sentence_idx + 1 < len(boundaries) - 1:
        next_end = boundaries[cite_sentence_idx + 2]
        after_sentence = paragraph[s_end:next_end].strip()
    else:
        after_sentence = ""

    return {
        "sentence": paragraph[s_start:s_end].strip(),
        "sentence_start": s_start,
        "sentence_end": s_end,
        "before_sentence": before_sentence,
        "after_sentence": after_sentence,
    }


def extract_masked_contexts(tex_path: str) -> list:
    if not tex_path:
        return []

    try:
        with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_content = f.read()
    except Exception:
        return []

    content = clean_tex_content(raw_content)

    section_positions = [
        (m.start(), m.group(1).strip())
        for m in re.finditer(
            r'\\(?:section|subsection|subsubsection)\{([^}]*)\}', content
        )
    ]

    paragraphs = split_into_paragraphs(content)

    cite_pattern = re.compile(
        r'(\\cite[tp]?\*?\s*(?:\[[^\]]*\])?\s*\{([^}]+)\})'
    )

    results = []

    for cite_match in cite_pattern.finditer(content):
        cite_pos = cite_match.start()
        cite_end = cite_match.end()
        cite_command = cite_match.group(1).strip()
        cite_keys = [k.strip() for k in cite_match.group(2).split(",")]

        containing_para = None
        for para_start, para_end, para_text in paragraphs:
            if para_start <= cite_pos < para_end:
                containing_para = (para_start, para_end, para_text)
                break

        if not containing_para:
            continue

        para_start, para_end, para_text = containing_para

        if len(para_text) < 50:
            continue
        if para_text.strip().startswith("\\begin{") or \
           para_text.strip().startswith("\\end{"):
            continue

        section = get_section_at_position(cite_pos, section_positions)

        sent_info = find_citation_sentence(
            para_text, cite_pos, cite_end, para_start
        )

        # Build masked context: 1 sentence before + [CITE_HERE] + 1 sentence after
        parts = []
        if sent_info["before_sentence"]:
            parts.append(sent_info["before_sentence"])
        parts.append("[CITE_HERE]")
        if sent_info["after_sentence"]:
            parts.append(sent_info["after_sentence"])
        masked_paragraph = " ".join(parts)

        line_num = content[:cite_pos].count("\n") + 1

        results.append({
            "paragraph": para_text,
            "citation_sentence": sent_info["sentence"],
            "cite_command": cite_command,
            "cite_keys": cite_keys,
            "before_citation": sent_info["before_sentence"],
            "after_citation": sent_info["after_sentence"],
            "masked_paragraph": masked_paragraph,
            "section": section,
            "line_number": line_num,
            "paragraph_length": len(para_text),
            "num_sentences_in_paragraph": len(
                re.findall(r'[.!?]\s', para_text)
            ) + 1,
        })

    return results


# ============================================================
# Per-paper processing and saving
# ============================================================

def process_and_save_one_paper(row, output_dir: str) -> dict:
    """
    Process a single paper and save results to disk immediately.
    Returns metadata dict or None on failure.
    """
    arxiv_id = row["arxiv_id"]
    clean_id = clean_arxiv_id(arxiv_id)
    if not clean_id:
        return None

    clean_id_safe = clean_id.replace("/", "_")

    # Extract LaTeX
    extract_dir = extract_source(row["source_path"], arxiv_id)
    main_tex = find_main_tex(extract_dir)
    if not main_tex:
        return None

    # Parse bib
    bib_entries = extract_bib_entries(extract_dir)

    # Extract masked contexts
    contexts = extract_masked_contexts(main_tex)
    if not contexts:
        return None

    # Enrich contexts with bib info
    for ctx in contexts:
        ctx["bib_entries"] = {
            k: bib_entries.get(k, {}) for k in ctx["cite_keys"]
        }

    # Build metadata
    metadata = {
        "arxiv_id": arxiv_id,
        "doi": row.get("doi"),
        "title": row.get("title"),
        "year": int(row.get("year", 0)),
        "venue": row.get("conference_name"),
        "num_contexts": len(contexts),
    }

    # Save to data/arxiv_masked/{clean_id}/
    paper_dir = os.path.join(output_dir, clean_id_safe)
    os.makedirs(paper_dir, exist_ok=True)

    with open(os.path.join(paper_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(os.path.join(paper_dir, "bib_entries.json"), "w") as f:
        json.dump(bib_entries, f, indent=2, default=str)

    with open(os.path.join(paper_dir, "contexts.json"), "w") as f:
        json.dump(contexts, f, indent=2, default=str)

    return metadata


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Citation Context Extraction with Level-2 Masking")
    print(f"Per-paper saving to: {OUTPUT_DIR}/{{arxiv_id}}/")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV)
    df_to_process = df[df["source_path"].notna()].copy()
    total = len(df_to_process)
    print(f"Papers with downloaded sources: {total}")

    os.makedirs(EXTRACT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check already processed — look for contexts.json in each subdir
    already_done = set()
    if os.path.exists(OUTPUT_DIR):
        for d in os.listdir(OUTPUT_DIR):
            if os.path.exists(os.path.join(OUTPUT_DIR, d, "contexts.json")):
                already_done.add(d)

    # Filter to only unprocessed papers
    to_process = []
    for idx, row in df_to_process.iterrows():
        clean_id = clean_arxiv_id(row["arxiv_id"])
        if clean_id and clean_id.replace("/", "_") not in already_done:
            to_process.append((idx, row))

    print(f"Already processed:   {len(already_done)}")
    print(f"To process:          {len(to_process)}")

    if not to_process:
        print("Nothing to do!")
    else:
        papers_ok = 0
        total_contexts = 0
        failed = 0

        pbar = tqdm(to_process, desc="Parsing", unit="paper")
        executor = ThreadPoolExecutor(max_workers=1)

        for idx, row in pbar:
            try:
                future = executor.submit(process_and_save_one_paper, row, OUTPUT_DIR)
                meta = future.result(timeout=TIMEOUT_SECONDS)

                if meta:
                    papers_ok += 1
                    total_contexts += meta["num_contexts"]
                else:
                    failed += 1
            except FuturesTimeoutError:
                failed += 1
            except Exception:
                failed += 1

            pbar.set_postfix(ok=papers_ok, contexts=total_contexts, failed=failed)

        executor.shutdown(wait=False)

        pbar.close()

    # --------------------------------------------------------
    # Summary (count from disk to be accurate)
    # --------------------------------------------------------
    final_papers = 0
    final_contexts = 0
    section_counts = {}

    for d in os.listdir(OUTPUT_DIR):
        meta_path = os.path.join(OUTPUT_DIR, d, "metadata.json")
        ctx_path = os.path.join(OUTPUT_DIR, d, "contexts.json")
        if not os.path.exists(meta_path) or not os.path.exists(ctx_path):
            continue

        final_papers += 1
        with open(meta_path) as f:
            meta = json.load(f)
        final_contexts += meta.get("num_contexts", 0)

        with open(ctx_path) as f:
            contexts = json.load(f)
        for ctx in contexts:
            sec = ctx.get("section", "Unknown")
            section_counts[sec] = section_counts.get(sec, 0) + 1

    print(f"\n{'=' * 70}")
    print("SUMMARY (from disk)")
    print(f"{'=' * 70}")
    print(f"  Total papers parsed:       {final_papers}")
    print(f"  Total citation contexts:   {final_contexts}")
    if final_papers:
        print(f"  Avg contexts per paper:    {final_contexts / final_papers:.1f}")

    if section_counts:
        print(f"\n  Section distribution (top 10):")
        for sec, cnt in sorted(section_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {sec}: {cnt}")

    # Show one example
    for d in sorted(os.listdir(OUTPUT_DIR))[:1]:
        ctx_path = os.path.join(OUTPUT_DIR, d, "contexts.json")
        meta_path = os.path.join(OUTPUT_DIR, d, "metadata.json")
        if os.path.exists(ctx_path) and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            with open(ctx_path) as f:
                contexts = json.load(f)
            if contexts:
                ex = contexts[0]
                print(f"\n  --- Example: {d} ---")
                print(f"  Paper: {meta.get('title', '')[:80]}")
                print(f"  Section: {ex['section']}")
                print(f"  Cite keys: {ex['cite_keys']}")
                print(f"\n  Citation sentence:")
                print(f"    {ex['citation_sentence'][:200]}")
                print(f"\n  Masked paragraph (LLM sees this):")
                print(f"    {ex['masked_paragraph'][:300]}")

    print(f"\n  Output: {OUTPUT_DIR}/")
    print(f"    ├── {{arxiv_id}}/metadata.json")
    print(f"    ├── {{arxiv_id}}/bib_entries.json")
    print(f"    └── {{arxiv_id}}/contexts.json")
    print()




#%%

#%%


"""
Script 02b: Analyze extracted citation contexts from arxiv_masked/
================================================================
1. Count papers where bib_entries.json is NOT empty ({})
2. Pointplot of avg num_contexts by year and by venue

Usage:
    python 02b_analyze_contexts.py
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['axes.grid'] = True  # Ensure grid lines are enabled
plt.rcParams['axes.grid.which'] = 'major'  # Apply grid lines only to major ticks
plt.rcParams['axes.grid.axis'] = 'y'  # Only show horizontal grid lines
plt.rcParams['grid.linestyle'] = '--'  # Set grid line style to dashed
plt.rcParams['grid.alpha'] = 0.3  # Set grid line transparency
# ============================================================
# CONFIG
# ============================================================

MASKED_DIR = "data/arxiv_masked"

import glob

# ============================================================
# Load all metadata (fast: glob + minimal I/O)
# ============================================================

print("Loading metadata...")
meta_files = glob.glob(os.path.join(MASKED_DIR, "*/metadata.json"))
print(f"  Found {len(meta_files)} paper directories")

records = []
bib_nonempty = 0
bib_empty = 0

for mf in meta_files:
    paper_dir = os.path.dirname(mf)

    with open(mf) as f:
        meta = json.load(f)

    # Fast bib check: just check file size instead of parsing JSON
    # An empty {} is 2 bytes; anything bigger has real entries
    bib_path = os.path.join(paper_dir, "bib_entries.json")
    if os.path.exists(bib_path) and os.path.getsize(bib_path) > 10:
        has_bib = True
        bib_nonempty += 1
    else:
        has_bib = False
        bib_empty += 1

    meta["has_bib"] = has_bib
    meta["folder"] = os.path.basename(paper_dir)
    records.append(meta)

df = pd.DataFrame(records)
total = len(df)

print("=" * 60)
print("CITATION CONTEXT ANALYSIS")
print("=" * 60)
print(f"  Total papers parsed:           {total}")
print(f"  Papers WITH bib entries:       {bib_nonempty}  ({100*bib_nonempty/max(total,1):.1f}%)")
print(f"  Papers WITHOUT bib entries:    {bib_empty}  ({100*bib_empty/max(total,1):.1f}%)")
print()

# ============================================================
# Derive venue from DOI
# ============================================================

def get_venue(doi):
    if pd.isna(doi) or not doi:
        return 'Unknown'
    doi = doi.lower()
    if 'findings-emnlp' in doi:
        return 'Findings-EMNLP'
    elif 'findings-acl' in doi:
        return 'Findings-ACL'
    elif 'findings-naacl' in doi:
        return 'Findings-NAACL'
    elif 'emnlp-main' in doi:
        return 'EMNLP Main'
    elif 'naacl-main' in doi or 'naacl-long' in doi or 'naacl-short' in doi:
        return 'NAACL Main'
    elif 'acl-main' in doi or 'acl-long' in doi or 'acl-short' in doi:
        return 'ACL Main'
    else:
        return 'Other'

df['venue_short'] = df['doi'].apply(get_venue)

venue_order = [
    'ACL Main', 'Findings-ACL',
    'EMNLP Main', 'Findings-EMNLP',
    'NAACL Main', 'Findings-NAACL',
]

palette = {
    'ACL Main': '#e41a1c',       'Findings-ACL': '#fb9a99',
    'EMNLP Main': '#377eb8',     'Findings-EMNLP': '#a6cee3',
    'NAACL Main': '#4daf4a',     'Findings-NAACL': '#b2df8a',
}

# Filter to known venues
df_plot = df[df['venue_short'].isin(venue_order)].copy()

print(f"  Papers in known venues:        {len(df_plot)}")
print(f"\n  By venue:")
for v in venue_order:
    cnt = (df_plot['venue_short'] == v).sum()
    avg_ctx = df_plot.loc[df_plot['venue_short'] == v, 'num_contexts'].mean()
    print(f"    {v:20s}: {cnt:5d} papers, avg {avg_ctx:.1f} contexts")

print(f"\n  By year:")
for y in sorted(df_plot['year'].unique()):
    cnt = (df_plot['year'] == y).sum()
    avg_ctx = df_plot.loc[df_plot['year'] == y, 'num_contexts'].mean()
    print(f"    {y}: {cnt:5d} papers, avg {avg_ctx:.1f} contexts")

# ============================================================
# Plot
# ============================================================

# Only papers with bib entries for plot 1
df_bib = df_plot[df_plot['has_bib'] == True].copy()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=300)

# --- Left: Count of papers with .bib by year & venue (grouped bar) ---
ax = axes[0]
years = sorted(df_plot['year'].unique())
n_venues = len(venue_order)
bar_width = 0.7 / n_venues
x = np.arange(len(years))

for i, v in enumerate(venue_order):
    counts = []
    for y in years:
        cnt = ((df_bib['venue_short'] == v) & (df_bib['year'] == y)).sum()
        counts.append(cnt)
    offset = (i - n_venues / 2 + 0.5) * bar_width
    ax.bar(x + offset, counts, width=bar_width, color=palette[v], label=v,
           edgecolor='white', linewidth=0.3)

ax.set_xlabel('Year', fontsize=10)
ax.set_ylabel('Paper Count (with .bib)', fontsize=10)
ax.set_title('Papers with .bib Entries by Year & Venue', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend(fontsize=6.5, ncol=2, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Right: Avg num_contexts pointplot by year & venue ---
ax = axes[1]
for v in venue_order:
    sub = df_plot[df_plot['venue_short'] == v]
    means = sub.groupby('year')['num_contexts'].mean()
    sems = sub.groupby('year')['num_contexts'].sem()
    valid_years = [y for y in years if y in means.index]
    ax.errorbar(
        valid_years,
        [means[y] for y in valid_years],
        yerr=[sems.get(y, 0) for y in valid_years],
        marker='o', markersize=5, capsize=3, linewidth=1.5,
        color=palette[v], label=v
    )

ax.set_xlabel('Year', fontsize=10)
ax.set_ylabel('Avg Citation Contexts per Paper', fontsize=10)
ax.set_title('Avg Contexts by Year & Venue', fontsize=11)
ax.set_xticks(years)
ax.legend(fontsize=6.5, ncol=2, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('data/context_analysis.png', bbox_inches='tight', dpi=300)
plt.show()
print(f"\n  Plot saved to: data/context_analysis.png")
# %%
