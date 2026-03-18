#%%
import os
import json
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['axes.grid'] = True  # Ensure grid lines are enabled
plt.rcParams['axes.grid.which'] = 'major'  # Apply grid lines only to major ticks
plt.rcParams['axes.grid.axis'] = 'y'  # Only show horizontal grid lines
plt.rcParams['grid.linestyle'] = '--'  # Set grid line style to dashed
plt.rcParams['grid.alpha'] = 0.3  # Set grid line transparency

#%%
df_preprints_dimensions = pd.read_csv('data/preprints_acl_dimensions.csv')

df_preprints_dimensions.head()


#%%
# Clean up conference names for better display
df = df_preprints_dimensions.copy()
df['venue_short'] = df['conference_name'].str.extract(
    r'(ACL|EMNLP|NAACL|Findings.*?(?:ACL|EMNLP|NAACL))'
)
# Simpler: just map based on DOI patterns
def get_venue(doi):
    if pd.isna(doi):
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

df['venue'] = df['doi'].apply(get_venue)

#%%
# Plot 1: Count by year
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=300)

# Plot 1: Count by year
sns.countplot(data=df, x='year', ax=axes[0], palette='viridis')
axes[0].set_title('Papers by Year', fontsize=14)
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Count')
for container in axes[0].containers:
    axes[0].bar_label(container)

# Plot 2: Count by year AND venue (grouped by conference)
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

sns.countplot(data=df, x='year', hue='venue', ax=axes[1],
              hue_order=venue_order, palette=palette)
axes[1].set_title('Papers by Year and Venue', fontsize=14)
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Count')
axes[1].legend(title='Venue', bbox_to_anchor=(1.05, 1), loc='upper left',
               fontsize=8, title_fontsize=9)

plt.tight_layout()
plt.savefig('data/papers_by_year_venue.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Print summary table
print(pd.crosstab(df['year'], df['venue'], margins=True))

#%%
# How many have arxiv IDs?
print(f"\nTotal papers: {len(df)}")
print(f"With arXiv ID: {df['arxiv_id'].notna().sum()} ({df['arxiv_id'].notna().mean():.1%})")
print(f"Without arXiv ID: {df['arxiv_id'].isna().sum()} ({df['arxiv_id'].isna().mean():.1%})")

# ArXiv coverage by year
print("\nArXiv coverage by year:")
print(df.groupby('year')['arxiv_id'].apply(lambda x: f"{x.notna().sum()}/{len(x)} ({x.notna().mean():.1%})"))

#%%

















# %%
"""
Script 1: Download arXiv LaTeX Sources
=======================================
Downloads LaTeX source (.tar.gz) files from arXiv for papers
that have arXiv IDs in your Dimensions export.

Run this first. It only downloads — no parsing.

Usage:
    python 01_download_arxiv_sources.py
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "data/preprints_acl_dimensions.csv"
DOWNLOAD_DIR = "data/arxiv_sources"

# arXiv rate: use export.arxiv.org, ~3 req/s is safe
DELAY_SECONDS = 0.3
MAX_WORKERS = 3


def clean_arxiv_id(arxiv_id: str) -> str:
    if pd.isna(arxiv_id):
        return None
    return arxiv_id.replace("arXiv:", "").replace("arxiv:", "").strip()


def download_one(arxiv_id: str, output_dir: str) -> str:
    clean_id = clean_arxiv_id(arxiv_id)
    if not clean_id:
        return None

    url = f"https://export.arxiv.org/e-print/{clean_id}"
    output_path = os.path.join(output_dir, f"{clean_id.replace('/', '_')}.tar.gz")

    if os.path.exists(output_path):
        return output_path

    try:
        resp = requests.get(url, timeout=30,
                            headers={"User-Agent": "CitationBiasResearch/1.0"})
        if resp.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return output_path
        elif resp.status_code == 429:
            time.sleep(5)
            resp = requests.get(url, timeout=30,
                                headers={"User-Agent": "CitationBiasResearch/1.0"})
            if resp.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                return output_path
        return None
    except Exception:
        return None


if __name__ == "__main__":
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df_with_arxiv = df[df["arxiv_id"].notna()].copy()
    total = len(df_with_arxiv)

    # Count already downloaded
    already = sum(
        1 for _, row in df_with_arxiv.iterrows()
        if os.path.exists(os.path.join(
            DOWNLOAD_DIR,
            f"{clean_arxiv_id(row['arxiv_id']).replace('/', '_')}.tar.gz"
        ))
    )

    print(f"Total papers:        {len(df)}")
    print(f"With arXiv ID:       {total}")
    print(f"Already downloaded:  {already}")
    print(f"To download:         {total - already}")

    source_paths = {}
    failed = 0

    def _worker(idx, arxiv_id):
        time.sleep(DELAY_SECONDS)
        return idx, download_one(arxiv_id, DOWNLOAD_DIR)

    pbar = tqdm(total=total, desc="Downloading", unit="paper")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_worker, idx, row["arxiv_id"]): idx
            for idx, row in df_with_arxiv.iterrows()
        }
        for future in as_completed(futures):
            idx, path = future.result()
            source_paths[idx] = path
            if path is None:
                failed += 1
            pbar.update(1)
            pbar.set_postfix(ok=len(source_paths) - failed, failed=failed)

    pbar.close()

    # Save download status back to CSV
    df["source_path"] = df.index.map(lambda i: source_paths.get(i))
    df.to_csv("data/preprints_with_source_paths.csv", index=False)

    success = sum(1 for v in source_paths.values() if v is not None)
    print(f"\nDone! Downloaded {success}, failed {failed}")
    print(f"Saved to: data/preprints_with_source_paths.csv")
# %%
