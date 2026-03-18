#%%

"""
Semantic Scholar API Demo
=========================
Retrieves paper metadata, references (with citation contexts & intents),
and citations for a given paper.

Usage:
    python semantic_scholar_demo.py

Notes:
    - Works WITHOUT an API key (unauthenticated, shared rate limit)
    - If you have an API key, set it below for 1 RPS dedicated rate
    - Key fields for citation context research:
        * contexts  — the sentence(s) surrounding the citation
        * intents   — classification: "background", "methodology", "result"
        * isInfluential — whether S2 considers it an influential citation
"""

import requests
import json
import time

#%%
# ============================================================
# CONFIGURATION
# ============================================================

# Set your API key here if you have one (otherwise leave as None)
API_KEY = None  # e.g., "your-api-key-here"

# Headers (with or without API key)
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}

# Base URL for Academic Graph API
BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Rate limiting: be polite to the API
DELAY = 1.0 if API_KEY else 3.0  # seconds between requests


def get_paper_details(paper_id: str) -> dict:
    """
    Get basic metadata for a paper.
    
    Accepts paper_id in formats like:
        - Semantic Scholar ID: "649def34f8be52c8b66281af98ae884c09aef38b"
        - ArXiv ID:           "ARXIV:2405.15739"
        - DOI:                "DOI:10.18653/v1/N19-1361"
        - ACL Anthology:      "ACL:N19-1361"
        - Corpus ID:          "CorpusId:215416146"
    """
    url = f"{BASE_URL}/paper/{paper_id}"
    
    fields = ",".join([
        "title",
        "abstract",
        "year",
        "venue",
        "publicationDate",
        "citationCount",
        "influentialCitationCount",
        "authors",
        "externalIds",          # DOI, ArXiv ID, etc.
        "s2FieldsOfStudy",      # field of study tags
        "publicationTypes",     # journal, conference, etc.
    ])
    
    params = {"fields": fields}
    response = requests.get(url, params=params, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"  Error fetching paper details: {response.status_code}")
        print(f"  {response.text}")
        return {}


def get_paper_references(paper_id: str, limit: int = 100) -> list:
    """
    Get papers REFERENCED BY the given paper (its bibliography).
    
    Crucially, this returns:
        - contexts: the citation context sentences from the CITING paper
        - intents:  S2's classification of why it was cited
        - isInfluential: whether S2 considers it influential
    """
    url = f"{BASE_URL}/paper/{paper_id}/references"
    
    # These are the key fields for your research!
    fields = ",".join([
        "title",
        "year",
        "venue",
        "authors",
        "citationCount",
        "externalIds",
        "contexts",        # <-- citation context sentences!
        "intents",         # <-- background / methodology / result
        "isInfluential",   # <-- influential citation flag
    ])
    
    all_references = []
    offset = 0
    
    while True:
        params = {
            "fields": fields,
            "offset": offset,
            "limit": min(limit - len(all_references), 1000),
        }
        
        response = requests.get(url, params=params, headers=HEADERS)
        
        if response.status_code != 200:
            print(f"  Error fetching references: {response.status_code}")
            break
        
        data = response.json()
        batch = data.get("data", [])
        all_references.extend(batch)
        
        # Check if there are more results
        if "next" in data and len(all_references) < limit:
            offset = data["next"]
            time.sleep(DELAY)
        else:
            break
    
    return all_references


def get_paper_citations(paper_id: str, limit: int = 50) -> list:
    """
    Get papers that CITE the given paper.
    
    Also returns contexts/intents showing HOW other papers cite this one.
    """
    url = f"{BASE_URL}/paper/{paper_id}/citations"
    
    fields = ",".join([
        "title",
        "year",
        "venue",
        "authors",
        "citationCount",
        "contexts",
        "intents",
        "isInfluential",
    ])
    
    params = {
        "fields": fields,
        "offset": 0,
        "limit": min(limit, 1000),
    }
    
    response = requests.get(url, params=params, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"  Error fetching citations: {response.status_code}")
        return []


# ============================================================
# MAIN DEMO
# ============================================================

if __name__ == "__main__":
    
    # Example: The SciCite paper by Cohan et al. (NAACL 2019)
    # You can use ArXiv IDs, DOIs, or S2 IDs
    PAPER_ID = "10.18653/v1/N19-1361"
    
    print("=" * 70)
    print("SEMANTIC SCHOLAR API DEMO")
    print("=" * 70)
    
    # ----------------------------------------------------------
    # 1. Get paper details
    # ----------------------------------------------------------
    print(f"\n[1] Fetching paper details for: {PAPER_ID}")
    paper = get_paper_details(PAPER_ID)
    
    if paper:
        print(f"\n  Title:    {paper.get('title')}")
        print(f"  Year:     {paper.get('year')}")
        print(f"  Venue:    {paper.get('venue')}")
        print(f"  Date:     {paper.get('publicationDate')}")
        print(f"  Citations: {paper.get('citationCount')}")
        print(f"  Influential Citations: {paper.get('influentialCitationCount')}")
        
        authors = paper.get("authors", [])
        print(f"  Authors ({len(authors)}):")
        for a in authors[:5]:  # Show first 5
            print(f"    - {a.get('name')} (ID: {a.get('authorId')})")
        
        fields_of_study = paper.get("s2FieldsOfStudy", [])
        if fields_of_study:
            categories = [f.get("category") for f in fields_of_study]
            print(f"  Fields:   {', '.join(set(categories))}")
        
        ext_ids = paper.get("externalIds", {})
        if ext_ids:
            print(f"  DOI:      {ext_ids.get('DOI', 'N/A')}")
            print(f"  ArXiv:    {ext_ids.get('ArXiv', 'N/A')}")
    
    time.sleep(DELAY)
    
    # ----------------------------------------------------------
    # 2. Get references WITH citation contexts
    # ----------------------------------------------------------
    print(f"\n[2] Fetching references (bibliography) with citation contexts...")
    references = get_paper_references(PAPER_ID, limit=50)
    print(f"  Found {len(references)} references")
    
    # Show a few examples with contexts
    print("\n  --- Example references with citation contexts ---")
    shown = 0
    for ref in references:
        cited_paper = ref.get("citedPaper", {})
        contexts = ref.get("contexts", [])
        intents = ref.get("intents", [])
        is_influential = ref.get("isInfluential", False)
        
        if contexts:  # Only show references that have context data
            print(f"\n  Referenced paper: {cited_paper.get('title', 'Unknown')}")
            print(f"  Year: {cited_paper.get('year')}  |  "
                  f"Venue: {cited_paper.get('venue', 'N/A')}  |  "
                  f"Citations: {cited_paper.get('citationCount', 'N/A')}")
            print(f"  Intents: {intents}")
            print(f"  Influential: {is_influential}")
            print(f"  Citation contexts ({len(contexts)}):")
            for i, ctx in enumerate(contexts[:2]):  # Show first 2 contexts
                # Truncate long contexts for display
                display = ctx[:200] + "..." if len(ctx) > 200 else ctx
                print(f"    [{i+1}] {display}")
            
            shown += 1
            if shown >= 3:
                break
    
    time.sleep(DELAY)
    
    # ----------------------------------------------------------
    # 3. Get citations (papers that cite this one)
    # ----------------------------------------------------------
    print(f"\n[3] Fetching citing papers (who cites this paper)...")
    citations = get_paper_citations(PAPER_ID, limit=10)
    print(f"  Found {len(citations)} citing papers (showing first 10)")
    
    print("\n  --- Example citing papers with contexts ---")
    shown = 0
    for cit in citations:
        citing_paper = cit.get("citingPaper", {})
        contexts = cit.get("contexts", [])
        intents = cit.get("intents", [])
        
        if contexts:
            print(f"\n  Citing paper: {citing_paper.get('title', 'Unknown')}")
            print(f"  Year: {citing_paper.get('year')}  |  "
                  f"Venue: {citing_paper.get('venue', 'N/A')}")
            print(f"  Intents: {intents}")
            print(f"  Contexts ({len(contexts)}):")
            for i, ctx in enumerate(contexts[:2]):
                display = ctx[:200] + "..." if len(ctx) > 200 else ctx
                print(f"    [{i+1}] {display}")
            
            shown += 1
            if shown >= 3:
                break
    
    # ----------------------------------------------------------
    # 4. Save everything to JSON for later analysis
    # ----------------------------------------------------------
    output = {
        "query_paper_id": PAPER_ID,
        "paper_details": paper,
        "references": references,
        "citations_sample": citations,
    }
    
    output_file = "s2_paper_data.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[4] Full data saved to: {output_file}")
    
    # ----------------------------------------------------------
    # Summary statistics
    # ----------------------------------------------------------
    refs_with_contexts = sum(
        1 for r in references if r.get("contexts")
    )
    total_contexts = sum(
        len(r.get("contexts", [])) for r in references
    )
    
    # Count intents
    intent_counts = {}
    for r in references:
        for intent in r.get("intents", []):
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total references:           {len(references)}")
    print(f"  References with contexts:   {refs_with_contexts}")
    print(f"  Total citation contexts:    {total_contexts}")
    print(f"  Intent distribution:        {intent_counts}")
    print(f"  Influential references:     "
          f"{sum(1 for r in references if r.get('isInfluential'))}")
    print()