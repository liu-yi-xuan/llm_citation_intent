#%%
"""
Script 03b: LLM-as-Judge Motivation Classification
==================================================
For each context from Script 03, ask an LLM to classify motivation for:
  1. The original citation sentence (ground truth)
  2. The LLM-filled citation sentence (generated in Script 03)

Both are judged using the same motivation guidelines:
supporting / contrasting / mentioning.

Saves results per-paper under:
    data/arxiv_original_llm_responses/{arxiv_id}/responses.json

Resume-safe: skips papers that already have responses.

Usage:
    python "03b llm_categorize_origin_citation.py"

Requires:
    pip install openai
    export OPENAI_API_KEY="sk-..."
"""

import os
import json
import glob
import time
import random
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIGURATION
# ============================================================

RESPONSES_DIR = "data/arxiv_llm_responses"      # Input: from script 03
OUTPUT_DIR = "data/arxiv_original_llm_responses"  # Output: LLM-as-judge classifications
MODEL = "deepseek/deepseek-chat-v3.1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
KEYS_FILE = "openrouter_keys.json"
MAX_WORKERS = 8
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0
MAX_PAPERS = None  # Set to int for testing (e.g., 5)


def _load_api_keys() -> list:
    """
    Load API keys with fallback priority:
      1) KEYS_FILE JSON: {"openrouter_api_keys": ["...", "..."]}
      2) OPENROUTER_API_KEYS (comma-separated)
      3) OPENROUTER_API_KEY
      4) OPENAI_API_KEY (fallback)
    """
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
            file_keys = payload.get("openrouter_api_keys", [])
            if isinstance(file_keys, list):
                keys = [k.strip() for k in file_keys if isinstance(k, str) and k.strip()]
                if keys:
                    return keys
        except Exception:
            pass

    keys_csv = os.getenv("OPENROUTER_API_KEYS", "").strip()
    if keys_csv:
        return [k.strip() for k in keys_csv.split(",") if k.strip()]

    single_or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if single_or_key:
        return [single_or_key]

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return [openai_key]

    return []


API_KEYS = _load_api_keys()
CLIENTS = [
    OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
    for key in API_KEYS
]

# ============================================================
# SYSTEM PROMPT — classify existing citation sentences
# ============================================================

SYSTEM_PROMPT = """You are an expert academic researcher in Natural Language Processing and Computational Linguistics.

Your task: Given a citation sentence from a research paper, classify the citation's motivation and your confidence in the classification.

You will be given:
- The title of the paper containing the citation
- The section where the citation appears
- The original citation sentence
- One sentence before the citation (context)
- One sentence after the citation (context)

Based on the citation sentence and its surrounding context, classify the citation motivation.

Respond ONLY with a valid JSON object (no markdown, no backticks, no explanation) in this exact format:

{
  "motivation": "One of: supporting, contrasting, mentioning",
  "motivation_explanation": "Brief explanation of why you chose this motivation category",
  "confidence": "high, medium, or low"
}

Guidelines for motivation:
- "supporting": The cited work provides evidence, methods, or findings that SUPPORT or ALIGN WITH the citing paper's claims or approach
- "contrasting": The cited work represents a COMPETING approach, CONTRADICTING finding, or a BASELINE that the citing paper improves upon or disagrees with
- "mentioning": The cited work is referenced for BACKGROUND context, definitions, or general acknowledgment without clear support or contrast"""


# ============================================================
# Call LLM API
# ============================================================

def classify_sentence(
    client: OpenAI,
    paper_title: str,
    section: str,
    citation_sentence: str,
    before_citation: str,
    after_citation: str,
) -> dict:
    """
    Send one citation sentence to the model for motivation classification.
    """
    user_prompt = f"""Paper title: "{paper_title}"
Section: {section}

Sentence before citation:
{before_citation if before_citation else "(beginning of paragraph)"}

Citation sentence:
{citation_sentence}

Sentence after citation:
{after_citation if after_citation else "(end of paragraph)"}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content.strip()
        result = json.loads(raw_text)

        result["_model"] = MODEL
        result["_usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return result

    except json.JSONDecodeError as e:
        return {
            "_error": "json_parse_error",
            "_raw_response": raw_text if 'raw_text' in dir() else "",
            "_error_detail": str(e),
        }
    except Exception as e:
        return {
            "_error": "api_error",
            "_error_detail": str(e),
        }


def classify_sentence_with_retry(
    client: OpenAI,
    paper_title: str,
    section: str,
    citation_sentence: str,
    before_citation: str,
    after_citation: str,
) -> dict:
    """
    Retry transient API failures with exponential backoff + jitter.
    """
    for attempt in range(MAX_RETRIES):
        result = classify_sentence(
            client=client,
            paper_title=paper_title,
            section=section,
            citation_sentence=citation_sentence,
            before_citation=before_citation,
            after_citation=after_citation,
        )
        if "_error" not in result:
            return result

        if result.get("_error") != "api_error":
            return result

        if attempt == MAX_RETRIES - 1:
            return result

        backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
        time.sleep(backoff)

    return {
        "_error": "api_error",
        "_error_detail": "retry_exhausted",
    }


# ============================================================
# Process one paper
# ============================================================

def process_one_paper(response_path: str, output_dir: str) -> dict:
    """
    Load Script 03 outputs and classify motivation for:
      - original citation sentence
      - LLM-filled citation sentence
    """
    with open(response_path) as f:
        data = json.load(f)

    arxiv_id = os.path.basename(os.path.dirname(response_path))
    paper_title = data.get("title", "")

    raw_results_03 = data.get("results", [])
    if not raw_results_03:
        return None

    # Keep one unique item per citation sentence context.
    seen = set()
    results_03 = []
    for r in raw_results_03:
        sig = (
            r.get("section", ""),
            r.get("citation_sentence_original", ""),
            r.get("before_citation", ""),
            r.get("after_citation", ""),
            r.get("masked_paragraph", ""),
        )
        if sig in seen:
            continue
        seen.add(sig)
        results_03.append(r)

    def _judge_one_context(i_r):
        i, r = i_r
        if "_error" in r.get("llm_response", {}):
            return i, None

        original_sentence = r.get("citation_sentence_original", "")
        llm_filled_sentence = r.get("llm_response", {}).get("citation_sentence", "")
        if not original_sentence and not llm_filled_sentence:
            return i, None

        client = CLIENTS[i % len(CLIENTS)]
        motivation_llm_as_judge_original = None
        motivation_llm_as_judge_filled = None

        if original_sentence:
            motivation_llm_as_judge_original = classify_sentence_with_retry(
                client=client,
                paper_title=paper_title,
                section=r.get("section", ""),
                citation_sentence=original_sentence,
                before_citation=r.get("before_citation", ""),
                after_citation=r.get("after_citation", ""),
            )

        if llm_filled_sentence:
            motivation_llm_as_judge_filled = classify_sentence_with_retry(
                client=client,
                paper_title=paper_title,
                section=r.get("section", ""),
                citation_sentence=llm_filled_sentence,
                before_citation=r.get("before_citation", ""),
                after_citation=r.get("after_citation", ""),
            )

        return i, {
            "context_index": r.get("context_index", i),
            "source_context_indices": r.get("source_context_indices", []),
            "section": r.get("section", ""),
            "cite_keys": r.get("cite_keys", []),
            "citation_sentence_original": original_sentence,
            "citation_sentence_filled": llm_filled_sentence,
            "masked_paragraph": r.get("masked_paragraph", ""),
            "before_citation": r.get("before_citation", ""),
            "after_citation": r.get("after_citation", ""),
            "num_citations_required": r.get("num_citations_required", max(1, len(r.get("cite_keys", [])))),
            "motivation_script03_self": {
                "motivation": r.get("llm_response", {}).get("motivation"),
                "motivation_explanation": r.get("llm_response", {}).get("motivation_explanation"),
                "confidence": r.get("llm_response", {}).get("confidence"),
            },
            "motivation_llm_as_judge_original": motivation_llm_as_judge_original,
            "motivation_llm_as_judge_filled": motivation_llm_as_judge_filled,
        }

    classifications_by_idx = {}
    workers = max(1, min(MAX_WORKERS, len(CLIENTS), len(results_03)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_judge_one_context, (i, r))
            for i, r in enumerate(results_03)
        ]
        for fut in as_completed(futures):
            i, record = fut.result()
            if record is not None:
                classifications_by_idx[i] = record

    classifications = [classifications_by_idx[i] for i in sorted(classifications_by_idx)]

    # Save
    out_dir = os.path.join(output_dir, arxiv_id)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "responses.json"), "w") as f:
        json.dump({
            "arxiv_id": data.get("arxiv_id"),
            "doi": data.get("doi"),
            "title": paper_title,
            "year": data.get("year"),
            "venue": data.get("venue"),
            "model": MODEL,
            "num_input_results": len(raw_results_03),
            "num_classified": len(classifications),
            "results": classifications,
        }, f, indent=2, default=str)

    return {
        "arxiv_id": arxiv_id,
        "num_classified": len(classifications),
        "num_errors": sum(
            1
            for c in classifications
            for k in [
                "motivation_llm_as_judge_original",
                "motivation_llm_as_judge_filled",
            ]
            if isinstance(c.get(k), dict) and "_error" in c[k]
        ),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not CLIENTS:
        raise RuntimeError(
            "No API keys found. Set openrouter_keys.json, OPENROUTER_API_KEYS, OPENROUTER_API_KEY, or OPENAI_API_KEY."
        )

    print("=" * 70)
    print("Classify Motivation (Original + Filled Citation Sentences)")
    print(f"Model: {MODEL}")
    print(f"Provider base URL: {OPENROUTER_BASE_URL}")
    print(f"Loaded API keys: {len(CLIENTS)}")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Input:  {RESPONSES_DIR}/{{arxiv_id}}/responses.json")
    print(f"Output: {OUTPUT_DIR}/{{arxiv_id}}/responses.json")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: Find all papers with LLM responses from script 03
    # --------------------------------------------------------
    response_files = sorted(glob.glob(
        os.path.join(RESPONSES_DIR, "*/responses.json")
    ))
    print(f"Papers with LLM responses: {len(response_files)}")

    # --------------------------------------------------------
    # Step 2: Check already processed
    # --------------------------------------------------------
    already_done = set()
    if os.path.exists(OUTPUT_DIR):
        already_done = {
            d for d in os.listdir(OUTPUT_DIR)
            if os.path.exists(os.path.join(OUTPUT_DIR, d, "responses.json"))
        }

    to_process = [
        rf for rf in response_files
        if os.path.basename(os.path.dirname(rf)) not in already_done
    ]

    print(f"Already classified:     {len(already_done)}")
    print(f"To classify:            {len(to_process)}")

    if MAX_PAPERS:
        to_process = to_process[:MAX_PAPERS]
        print(f"  (capped to {MAX_PAPERS} papers for testing)")

    # Estimate cost (OpenRouter pricing)
    # Pricing requested:
    #   input  = $0.15 / 1M tokens
    #   output = $0.75 / 1M tokens
    # Script 03b can make up to 2 calls per context:
    #   - judge original sentence
    #   - judge filled sentence
    avg_unique_contexts_per_paper = 30
    avg_calls_per_context = 2
    est_calls = len(to_process) * avg_unique_contexts_per_paper * avg_calls_per_context

    avg_input_tokens_per_call = 260
    avg_output_tokens_per_call = 120
    est_input_tokens = est_calls * avg_input_tokens_per_call
    est_output_tokens = est_calls * avg_output_tokens_per_call
    est_input_cost = est_input_tokens / 1_000_000 * 0.15
    est_output_cost = est_output_tokens / 1_000_000 * 0.75
    est_total_cost = est_input_cost + est_output_cost

    print(f"\n  Estimated API calls:   ~{est_calls:,}")
    print(f"  Estimated input tokens:~{est_input_tokens:,}")
    print(f"  Estimated output tokens:~{est_output_tokens:,}")
    print(f"  Estimated input cost:  ~${est_input_cost:.2f}")
    print(f"  Estimated output cost: ~${est_output_cost:.2f}")
    print(f"  Estimated total cost:  ~${est_total_cost:.2f} (rough)")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    papers_ok = 0
    total_classified = 0
    total_errors = 0

    pbar = tqdm(to_process, desc="Classifying", unit="paper")
    for rf in pbar:
        try:
            summary = process_one_paper(rf, OUTPUT_DIR)
            if summary:
                papers_ok += 1
                total_classified += summary["num_classified"]
                total_errors += summary["num_errors"]
        except Exception:
            pass

        pbar.set_postfix(
            ok=papers_ok,
            classified=total_classified,
            errors=total_errors
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Papers classified:      {papers_ok}")
    print(f"  Total sentences:        {total_classified}")
    print(f"  API errors:             {total_errors}")
    if papers_ok:
        print(f"  Avg sentences/paper:    {total_classified / papers_ok:.1f}")
    print(f"\n  Output: {OUTPUT_DIR}/")

#%%
