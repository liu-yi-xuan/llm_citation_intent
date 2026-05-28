#%%
"""
Step 02 — LLM-as-Judge intent classification (two-phase)
========================================================
For each citation context, classify intent as supporting / contrasting /
mentioning. Separated
into two phases so the original-side judgment is computed only ONCE per judge
instead of duplicated per source-model run.

PHASE A — Judge the ORIGINAL sentence (run once per judge)
  Iterates: data/arxiv_masked_2025/{arxiv_id}/contexts.json (all venues, all
  3,048 papers in scope; venue filter applies only to Phase B).
  Saves:    data/llm_responses_judge/original__{judge_slug}/{arxiv_id}/responses.json

PHASE B — Judge the FILLED sentence (run per source × judge pair)
  Iterates: data/arxiv_llm_responses/{source_slug}/{arxiv_id}/responses.json
  Saves:    data/llm_responses_judge/source__{source_slug}__judge__{judge_slug}/{arxiv_id}/responses.json
  Each Phase B file embeds the matching Phase A judgment so the file is fully
  self-contained for downstream analysis (Step 5 / Dimensions matching).

Per-context output schema (Phase B example):
  {
    "context_index": 17, "section": "...",
    "cite_keys": [...], "cite_commands": [...], "bib_entries": {...},
    "citation_sentence_original": "...", "citation_sentence_filled": "...",
    "before_citation": "...", "after_citation": "...", "masked_paragraph": "...",
    "num_citations_required": 6, "num_citations_returned": 6,
    "recommended_papers": [...],            # for Step 5 Dimensions matching
    "motivation_self": {...},               # source model self-label from Step 01
    "motivation_judge_original": {...},     # copied from Phase A
    "motivation_judge_filled": {...}        # newly computed
  }

Provider: OpenRouter (https://openrouter.ai/api/v1).
Keys:     openrouter_keys.json or env OPENROUTER_API_KEYS / OPENROUTER_API_KEY.

Default judge: google/gemini-2.0-flash-001 (cheapest reliable; 0.10% error
rate in our Script-03 run; ~$22 total to judge 6 source models on Main).

Usage:
    python "04 llm_judge_motivation.py"           # runs Phase A + all Phase Bs
    PHASE=A python "04 llm_judge_motivation.py"   # Phase A only
    PHASE=B python "04 llm_judge_motivation.py"   # Phase B only (assumes A done)

Resume-safe: per-context fix-up retries any contexts whose llm_response carries
an _error field. Per-paper resume skips arxiv_ids whose responses.json exists
and has no errors.

Requires: pip install openai pandas tqdm
"""

import os
import json
import glob
import time
import random
import re
import logging
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# CONFIGURATION
# ============================================================

MASKED_DIR = "data/arxiv_masked_2025"
RESPONSES_DIR = "data/arxiv_llm_responses"
# Output base dir is overridable via env var so two judges can run in parallel
# without colliding (e.g., arxiv_llm_judge_deepseek vs arxiv_llm_judge_gemini).
JUDGE_OUTPUT_DIR = os.getenv("JUDGE_OUTPUT_DIR", "data/llm_responses_judge")
PREPRINT_DATE_CSV = "data/preprints_acl_dimensions.csv"
ARXIV_DATES_CSV = "data/arxiv_api_dates.csv"
LOG_DIR = "logs"
LOG_TAG = "04_judge"

# Provider — configurable via env vars so the same script can target
# OpenRouter or SiliconFlow without changing source. Defaults to OpenRouter.
# To switch to SiliconFlow, set:
#   PROVIDER_BASE_URL=https://api.siliconflow.com/v1
#   KEYS_FILE=siliconflow_keys.json
#   KEYS_JSON_FIELD=siliconflow_api_keys
#   ENV_KEYS_PLURAL=SILICONFLOW_API_KEYS
#   ENV_KEYS_SINGULAR=SILICONFLOW_API_KEY
PROVIDER_BASE_URL = os.getenv("PROVIDER_BASE_URL", "https://openrouter.ai/api/v1")
KEYS_FILE = os.getenv("KEYS_FILE", "openrouter_keys.json")
KEYS_JSON_FIELD = os.getenv("KEYS_JSON_FIELD", "openrouter_api_keys")
ENV_KEYS_PLURAL = os.getenv("ENV_KEYS_PLURAL", "OPENROUTER_API_KEYS")
ENV_KEYS_SINGULAR = os.getenv("ENV_KEYS_SINGULAR", "OPENROUTER_API_KEY")

# Default judge: cheapest non-reasoning model with proven JSON discipline.
DEFAULT_JUDGE_MODEL = "google/gemini-2.0-flash-001"

# Source models whose Phase B (filled-sentence) judgments to compute.
# Each becomes a folder under data/arxiv_llm_responses/{slug}/
DEFAULT_SOURCE_MODELS = [
    "deepseek-ai/DeepSeek-V3.2",
    "Qwen/Qwen2.5-72B-Instruct",
    "openai/gpt-5.1-chat",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-4-maverick",
    "anthropic/claude-3.5-haiku",
]

# Filter for Phase B (per-source-model fillet judgments). Phase A scope is
# always the full dataset to maximize cache reuse.
INCLUDE_VENUES_PHASE_B = {"ACL Main", "EMNLP Main", "NAACL Main"}
PREPRINT_DATE_CUTOFF = "2024-12-31"
PUBLICATION_YEAR = 2025

# Run-time settings
MAX_WORKERS = 24
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0
MAX_PAPERS_PHASE_A = None     # set int for testing
MAX_PAPERS_PHASE_B = None
PHASE = os.getenv("PHASE", "AB").upper()  # "A", "B", or "AB"

# Window sizes (chars) for cleaned context
BEFORE_AFTER_MAX_CHARS = 500
SENTENCE_MAX_CHARS = 800
MAX_CITED_TITLES = 5


# ============================================================
# HELPERS
# ============================================================

def sanitize_model_slug(model_slug: str) -> str:
    out = []
    for ch in str(model_slug):
        out.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(out).strip("_")


def clean_arxiv_id(arxiv_id) -> str:
    if pd.isna(arxiv_id):
        return ""
    return str(arxiv_id).replace("arXiv:", "").replace("arxiv:", "").strip()


def arxiv_folder_name(arxiv_id) -> str:
    return clean_arxiv_id(arxiv_id).replace("/", "_")


def parse_models_env(env_name: str, default_list: list[str]) -> list[str]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return list(default_list)
    return [m.strip() for m in raw.split(",") if m.strip()]


def load_preprint_date_lookup(csv_path: str) -> dict:
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if "arxiv_id" not in df.columns or "preprint_date" not in df.columns:
        return {}
    out = {}
    for _, row in df[["arxiv_id", "preprint_date"]].dropna(subset=["arxiv_id"]).iterrows():
        folder = arxiv_folder_name(row["arxiv_id"])
        if folder:
            out[folder] = row.get("preprint_date")
    return out


def load_arxiv_date_lookup(csv_path: str) -> dict:
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    required = {"arxiv_id_clean", "first_arxiv_date", "latest_arxiv_date"}
    if not required.issubset(df.columns):
        return {}
    out = {}
    for _, row in df[list(required)].dropna(subset=["arxiv_id_clean"]).iterrows():
        folder = arxiv_folder_name(row["arxiv_id_clean"])
        if folder:
            out[folder] = {
                "first_arxiv_date": row.get("first_arxiv_date"),
                "latest_arxiv_date": row.get("latest_arxiv_date"),
            }
    return out


def get_venue(doi) -> str:
    if doi is None:
        return "Unknown"
    if isinstance(doi, float) and pd.isna(doi):
        return "Unknown"
    d = str(doi).strip().lower()
    if not d:
        return "Unknown"
    if "findings-emnlp" in d:    return "Findings-EMNLP"
    if "findings-acl" in d:      return "Findings-ACL"
    if "findings-naacl" in d:    return "Findings-NAACL"
    if "emnlp-main" in d:        return "EMNLP Main"
    if "naacl-main" in d or "naacl-long" in d or "naacl-short" in d: return "NAACL Main"
    if "acl-main" in d or "acl-long" in d or "acl-short" in d:       return "ACL Main"
    return "Other"


def _load_api_keys() -> list:
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
            file_keys = payload.get(KEYS_JSON_FIELD) or payload.get("api_keys") or []
            if isinstance(file_keys, list):
                keys = [k.strip() for k in file_keys if isinstance(k, str) and k.strip()]
                if keys:
                    return keys
        except Exception:
            pass
    csv_keys = os.getenv(ENV_KEYS_PLURAL, "").strip()
    if csv_keys:
        return [k.strip() for k in csv_keys.split(",") if k.strip()]
    single = os.getenv(ENV_KEYS_SINGULAR, "").strip()
    if single:
        return [single]
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return [openai_key]
    return []


API_KEYS = _load_api_keys()
CLIENTS = [OpenAI(api_key=k, base_url=PROVIDER_BASE_URL) for k in API_KEYS]


# ============================================================
# Logging + progress snapshot
# ============================================================

def setup_logger(tag: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{LOG_TAG}_run_{tag}.log")
    logger = logging.getLogger(f"script04.{tag}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    logger.info("=" * 60)
    logger.info(f"Run started at {datetime.now().isoformat(timespec='seconds')}")
    return logger, log_path


def write_snapshot(tag: str, state: dict) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    snap = os.path.join(LOG_DIR, f"{LOG_TAG}_progress_{tag}.json")
    tmp = snap + ".tmp"
    payload = {**state, "snapshot_time": datetime.now().isoformat(timespec="seconds")}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, snap)
    return snap


# ============================================================
# LaTeX cleaner (mirrors the reconstruction step)
# ============================================================

def clean_paragraph_for_llm(text: str, max_chars: int = 600) -> str:
    if not text:
        return text
    text = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "[TABLE]", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}", "[TABLE]", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "[FIGURE]", text, flags=re.DOTALL)
    text = re.sub(
        r"\\begin\{(equation|align|gather|eqnarray|multline|displaymath)\*?\}.*?\\end\{\1\*?\}",
        "[EQUATION]", text, flags=re.DOTALL,
    )
    text = re.sub(r"\$\$.*?\$\$", "[MATH]", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", "[MATH]", text, flags=re.DOTALL)
    def _inline_math(m):
        return "[MATH]" if len(m.group(1)) > 10 else ""
    text = re.sub(r"\$([^$\n]{1,200})\$", _inline_math, text)
    text = re.sub(
        r"\\(textbf|textit|emph|texttt|underline|textrm|textsc|sys|lm|data)\{([^{}]*)\}",
        r"\2", text,
    )
    text = re.sub(
        r"\\(section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^{}]*)\}",
        r"\2", text,
    )
    text = re.sub(r"\\(label|ref|cref|eqref|pageref|autoref)\{[^{}]*\}", "", text)
    for _ in range(2):
        text = re.sub(r"\\footnote\{[^{}]*\}", "", text)
    text = re.sub(r"\\footnote\{.*?\}(?=\s|$|[.,;:])", "", text, flags=re.DOTALL)
    text = re.sub(r"\\footnotemark\b", "", text)
    text = re.sub(r"\\url\{[^{}]*\}", "", text)
    text = re.sub(r"\\href\{[^{}]*\}\{[^{}]*\}", "", text)
    text = re.sub(
        r"\\(cite|citep|citet|citealp|citealt|citeauthor|citeyear|citeyearpar|nocite|fullcite|newcite)\*?\{[^{}]*\}",
        "[CITE]", text,
    )
    text = re.sub(
        r"\\(setlength|tabcolsep|toprule|midrule|bottomrule|cmidrule|scriptsize|"
        r"footnotesize|small|normalsize|large|Large|LARGE|huge|Huge|newline|"
        r"newpage|hline|noindent|centering|caption|multirow|multicolumn|"
        r"checkmark|ul|item|begin|end|forall|exists|sum|prod|int|frac)\b[^\n]*",
        "", text,
    )
    text = re.sub(r"\\[a-zA-Z]+\*?\{[^{}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


# ============================================================
# Judge prompt
# ============================================================

SYSTEM_PROMPT = """You classify citations in NLP papers as supporting, contrasting, or mentioning, based on a citation sentence and its surrounding context.

Categories:
- supporting: cited work provides evidence, methods, or findings aligned with the citing paper's claims or approach.
- contrasting: cited work is a competing approach, contradicting finding, or baseline the citing paper improves upon or disagrees with.
- mentioning: cited work is referenced for background, definitions, or general acknowledgment with no clear support or contrast.

Confidence:
- high: the surrounding text explicitly signals the relationship.
- medium: the relationship is strongly implied but not explicit.
- low: the relationship is ambiguous or inferable only with effort.

Return ONLY a valid JSON object, no markdown or text outside the JSON:

{
  "motivation": "supporting" | "contrasting" | "mentioning",
  "confidence": "high" | "medium" | "low"
}
"""


def parse_json_object_from_text(raw_text: str) -> dict:
    text = (raw_text or "").strip()
    if not text:
        raise json.JSONDecodeError("empty response", text, 0)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    return json.loads(text[start:end + 1])


def create_chat_completion(client, model: str, messages: list, max_tokens: int):
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0,
            max_tokens=max_tokens, response_format={"type": "json_object"},
        )
        return response, True
    except Exception:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, max_tokens=max_tokens,
        )
        return response, False


def classify_sentence(client, judge_model, paper_title, section,
                       citation_sentence, before_citation, after_citation) -> dict:
    """Classify motivation based ONLY on the sentence + local rhetorical context.
    Cited paper titles are intentionally NOT passed: motivation is a property of
    the rhetorical move (verb choice, discourse markers, section position),
    independent of what specific work is cited. Keeping this clean also makes
    Phase A (original) and Phase B (filled) symmetric — the judge sees the same
    inputs except for the sentence itself.
    """
    cleaned_before = clean_paragraph_for_llm(before_citation, max_chars=BEFORE_AFTER_MAX_CHARS)
    cleaned_after = clean_paragraph_for_llm(after_citation, max_chars=BEFORE_AFTER_MAX_CHARS)
    cleaned_sentence = clean_paragraph_for_llm(citation_sentence, max_chars=SENTENCE_MAX_CHARS)
    parts = [
        f'Paper: "{paper_title}"',
        f"Section: {section}" if section else "",
        "",
        f"Sentence before:\n{cleaned_before or '(beginning of paragraph)'}",
        "",
        f"Citation sentence:\n{cleaned_sentence}",
        "",
        f"Sentence after:\n{cleaned_after or '(end of paragraph)'}",
    ]
    user_prompt = "\n".join(p for p in parts if p)

    raw_text = ""
    try:
        response, used_json_mode = create_chat_completion(
            client=client, model=judge_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            # Generous cap so reasoning models (gemini-3-flash, deepseek-v4-flash)
            # have headroom for hidden reasoning tokens; visible JSON is ~25
            # tokens. DeepSeek-V4-Flash uses ~110 hidden reasoning tokens by
            # default and may need more for ambiguous cases; cap raised to 700
            # to give it room. We're billed for actual usage, not the cap.
            max_tokens=700,
        )
        raw_text = response.choices[0].message.content.strip()
        result = parse_json_object_from_text(raw_text)
        result["_model"] = judge_model
        result["_json_mode_enforced"] = used_json_mode
        result["_usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        result["_raw_response"] = raw_text
        result["_judged_at"] = datetime.now().isoformat(timespec="seconds")
        return result
    except json.JSONDecodeError as e:
        return {"_error": "json_parse_error", "_raw_response": raw_text, "_error_detail": str(e)}
    except Exception as e:
        return {"_error": "api_error", "_error_detail": str(e)}


def classify_with_retry(clients, idx, judge_model, paper_title, section,
                          citation_sentence, before_citation, after_citation) -> dict:
    for attempt in range(MAX_RETRIES):
        client = clients[(idx + attempt) % len(clients)]
        result = classify_sentence(
            client=client, judge_model=judge_model,
            paper_title=paper_title, section=section,
            citation_sentence=citation_sentence,
            before_citation=before_citation, after_citation=after_citation,
        )
        if "_error" not in result:
            return result
        # Retry api_error and json_parse_error (gemini-style, since judge prompt is small)
        if result.get("_error") not in {"api_error", "json_parse_error"}:
            return result
        if attempt == MAX_RETRIES - 1:
            return result
        backoff = min(MAX_BACKOFF_SECONDS, INITIAL_BACKOFF_SECONDS * (2 ** attempt)) + random.uniform(0, 1.0)
        time.sleep(backoff)
    return {"_error": "api_error", "_error_detail": "retry_exhausted"}


# ============================================================
# Context dedup (mirrors 03's merge_contexts_by_sentence)
# ============================================================

def context_signature(ctx: dict) -> tuple:
    return (
        ctx.get("section", ""),
        ctx.get("citation_sentence", ""),
        ctx.get("before_citation", ""),
        ctx.get("after_citation", ""),
        ctx.get("masked_paragraph", ""),
    )


def merge_raw_contexts(contexts: list) -> list:
    """Merge contexts that share sentence-level signature; aggregate cite_keys."""
    grouped = {}
    for raw_idx, ctx in enumerate(contexts):
        sig = context_signature(ctx)
        if sig not in grouped:
            grouped[sig] = {
                "section": ctx.get("section", ""),
                "citation_sentence_original": ctx.get("citation_sentence", ""),
                "masked_paragraph": ctx.get("masked_paragraph", ""),
                "before_citation": ctx.get("before_citation", ""),
                "after_citation": ctx.get("after_citation", ""),
                "cite_commands": [],
                "cite_keys": [],
                "source_context_indices": [],
                "bib_entries": {},
            }
        bucket = grouped[sig]
        bucket["source_context_indices"].append(raw_idx)
        bucket["cite_commands"].append(ctx.get("cite_command", ""))
        for key in ctx.get("cite_keys", []):
            if key not in bucket["cite_keys"]:
                bucket["cite_keys"].append(key)
        for k, v in ctx.get("bib_entries", {}).items():
            if k not in bucket["bib_entries"]:
                bucket["bib_entries"][k] = v
    merged = list(grouped.values())
    for item in merged:
        item["num_required_citations"] = max(1, len(item["cite_keys"]))
    return merged


def extract_cited_titles(cite_keys: list, bib_entries: dict) -> list[str]:
    titles = []
    for ck in cite_keys or []:
        entry = (bib_entries or {}).get(ck) or {}
        t = str(entry.get("title", "") or "").strip()
        if t and t not in titles:
            titles.append(t)
    return titles


# ============================================================
# Phase A — judge ORIGINAL sentences (per arxiv paper)
# ============================================================

def phase_a_dir(judge_model: str) -> str:
    return os.path.join(JUDGE_OUTPUT_DIR, f"original__{sanitize_model_slug(judge_model)}")


def phase_a_paper_path(judge_model: str, arxiv_id: str) -> str:
    return os.path.join(phase_a_dir(judge_model), arxiv_id, "responses.json")


def phase_a_paper_done(judge_model: str, arxiv_id: str) -> bool:
    """A paper is 'done' for Phase A iff the file exists and every record has a
    valid motivation_judge_original (no _error)."""
    fp = phase_a_paper_path(judge_model, arxiv_id)
    if not os.path.exists(fp):
        return False
    try:
        with open(fp) as f:
            data = json.load(f)
    except Exception:
        return False
    for r in data.get("results", []):
        m = r.get("motivation_judge_original") or {}
        if not isinstance(m, dict) or "_error" in m or not m.get("motivation"):
            return False
    return True


def run_phase_a_one_paper(meta: dict, contexts_path: str, bib_path: str,
                          judge_model: str, run_logger) -> dict | None:
    """Process one paper for Phase A. Writes the per-paper file. Returns summary."""
    arxiv_id = meta["folder"]
    paper_title = meta.get("title", "")
    paper_year = meta.get("year")
    paper_venue = meta.get("venue", "")

    if not os.path.exists(contexts_path):
        return None
    with open(contexts_path) as f:
        raw_contexts = json.load(f)
    if not raw_contexts:
        return None

    unique_contexts = merge_raw_contexts(raw_contexts)

    # Load any existing file so we can preserve already-judged contexts.
    out_path = phase_a_paper_path(judge_model, arxiv_id)
    existing = {}  # sig -> motivation dict
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                prev = json.load(f)
            for r in prev.get("results", []):
                key = (r.get("section",""), r.get("citation_sentence_original",""))
                m = r.get("motivation_judge_original")
                if isinstance(m, dict) and "_error" not in m and m.get("motivation"):
                    existing[key] = m
        except Exception:
            pass

    # Per-context worker (judge only the originals that are missing or errored)
    def _judge_one(i_ctx):
        i, ctx = i_ctx
        key = (ctx.get("section",""), ctx.get("citation_sentence_original",""))
        if key in existing:
            return i, existing[key]
        result = classify_with_retry(
            clients=CLIENTS, idx=i, judge_model=judge_model,
            paper_title=paper_title, section=ctx.get("section", ""),
            citation_sentence=ctx.get("citation_sentence_original", ""),
            before_citation=ctx.get("before_citation", ""),
            after_citation=ctx.get("after_citation", ""),
        )
        return i, result

    motivations = {}
    workers = max(1, min(MAX_WORKERS, len(unique_contexts)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_judge_one, (i, ctx)) for i, ctx in enumerate(unique_contexts)]
        for fut in as_completed(futures):
            i, m = fut.result()
            motivations[i] = m

    # Build results in original context order
    results = []
    for i, ctx in enumerate(unique_contexts):
        results.append({
            "context_index": i,
            "source_context_indices": ctx.get("source_context_indices", []),
            "section": ctx.get("section", ""),
            "cite_keys": ctx.get("cite_keys", []),
            "cite_commands": ctx.get("cite_commands", []),
            "bib_entries": ctx.get("bib_entries", {}),
            "citation_sentence_original": ctx.get("citation_sentence_original", ""),
            "before_citation": ctx.get("before_citation", ""),
            "after_citation": ctx.get("after_citation", ""),
            "masked_paragraph": ctx.get("masked_paragraph", ""),
            "num_citations_required": ctx.get("num_required_citations", 1),
            "motivation_judge_original": motivations.get(i, {"_error": "missing"}),
        })

    out_data = {
        "arxiv_id": meta.get("arxiv_id"),
        "doi": meta.get("doi"),
        "title": paper_title,
        "year": meta.get("year"),
        "venue": paper_venue,
        "preprint_date": meta.get("preprint_date"),
        "first_arxiv_date": meta.get("first_arxiv_date"),
        "latest_arxiv_date": meta.get("latest_arxiv_date"),
        "judge_model": judge_model,
        "num_raw_contexts": len(raw_contexts),
        "num_contexts": len(results),
        "results": results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, default=str)
    os.replace(tmp, out_path)

    n_err = sum(1 for r in results if "_error" in (r["motivation_judge_original"] or {}))
    if run_logger:
        run_logger.info(
            f"PHASE_A {arxiv_id}: contexts={len(results)} errors={n_err}"
        )
    return {"arxiv_id": arxiv_id, "num_contexts": len(results), "num_errors": n_err}


def run_phase_a(judge_model: str, all_papers: list[dict], run_logger):
    """Iterate the full paper set, process each via run_phase_a_one_paper."""
    todo = [m for m in all_papers if not phase_a_paper_done(judge_model, m["folder"])]
    if MAX_PAPERS_PHASE_A:
        todo = todo[:MAX_PAPERS_PHASE_A]
    print(f"\n--- PHASE A — judge ORIGINAL sentences ---")
    print(f"Judge model     : {judge_model}")
    print(f"Output dir      : {phase_a_dir(judge_model)}")
    print(f"Total papers    : {len(all_papers)}")
    print(f"Already done    : {len(all_papers) - len(todo)}")
    print(f"To judge        : {len(todo)}")
    if run_logger:
        run_logger.info(f"PHASE A start: total={len(all_papers)} todo={len(todo)}")

    if not todo:
        return

    started = time.time()
    papers_ok = total_ctx = total_err = 0
    pair_slug = f"original__{sanitize_model_slug(judge_model)}"
    write_snapshot(pair_slug, {
        "phase": "A", "judge_model": judge_model,
        "total_papers": len(all_papers), "todo": len(todo),
        "papers_done_this_run": 0, "elapsed_seconds": 0, "status": "starting",
    })

    pbar = tqdm(todo, desc=f"Phase A ({sanitize_model_slug(judge_model)})", unit="paper")
    for idx, meta in enumerate(pbar):
        contexts_path = os.path.join(meta["paper_dir"], "contexts.json")
        bib_path = os.path.join(meta["paper_dir"], "bib_entries.json")
        try:
            summary = run_phase_a_one_paper(meta, contexts_path, bib_path, judge_model, run_logger)
            if summary:
                papers_ok += 1
                total_ctx += summary["num_contexts"]
                total_err += summary["num_errors"]
        except Exception as e:
            if run_logger:
                run_logger.exception(f"PHASE_A FAIL {meta['folder']}: {e}")
        elapsed = time.time() - started
        pbar.set_postfix(ok=papers_ok, ctx=total_ctx, err=total_err)

        if (idx + 1) % 5 == 0 or idx == len(todo) - 1:
            ppm = (idx + 1) / elapsed * 60 if elapsed > 0 else 0
            eta = (len(todo) - idx - 1) / ((idx + 1) / elapsed) if (idx + 1) and elapsed > 0 else None
            write_snapshot(pair_slug, {
                "phase": "A", "judge_model": judge_model,
                "total_papers": len(all_papers), "todo": len(todo),
                "papers_done_this_run": idx + 1,
                "papers_ok": papers_ok,
                "total_contexts": total_ctx, "total_errors": total_err,
                "current_paper": meta["folder"],
                "elapsed_seconds": round(elapsed, 1),
                "papers_per_minute": round(ppm, 2),
                "eta_seconds": round(eta, 0) if eta else None,
                "status": "running" if idx + 1 < len(todo) else "done",
            })

    if run_logger:
        run_logger.info(f"PHASE A done: papers_ok={papers_ok} ctx={total_ctx} err={total_err}")
    print(f"PHASE A summary: papers_ok={papers_ok} contexts={total_ctx} errors={total_err}")


# ============================================================
# Phase B — judge FILLED sentences (per source × judge × paper)
# ============================================================

def phase_b_dir(source_model: str, judge_model: str) -> str:
    return os.path.join(
        JUDGE_OUTPUT_DIR,
        f"source__{sanitize_model_slug(source_model)}__judge__{sanitize_model_slug(judge_model)}",
    )


def phase_b_paper_path(source_model: str, judge_model: str, arxiv_id: str) -> str:
    return os.path.join(phase_b_dir(source_model, judge_model), arxiv_id, "responses.json")


def phase_b_paper_done(source_model: str, judge_model: str, arxiv_id: str) -> bool:
    """A paper is 'done' for Phase B iff the file exists and every record has a
    valid motivation_judge_filled (no _error)."""
    fp = phase_b_paper_path(source_model, judge_model, arxiv_id)
    if not os.path.exists(fp):
        return False
    try:
        with open(fp) as f:
            data = json.load(f)
    except Exception:
        return False
    for r in data.get("results", []):
        m = r.get("motivation_judge_filled") or {}
        if not isinstance(m, dict) or "_error" in m or not m.get("motivation"):
            return False
    return True


def load_phase_a_motivations(judge_model: str, arxiv_id: str) -> dict:
    """Return {(section, citation_sentence_original): motivation_dict} from Phase A file."""
    fp = phase_a_paper_path(judge_model, arxiv_id)
    out = {}
    if not os.path.exists(fp):
        return out
    try:
        with open(fp) as f:
            data = json.load(f)
    except Exception:
        return out
    for r in data.get("results", []):
        key = (r.get("section", ""), r.get("citation_sentence_original", ""))
        m = r.get("motivation_judge_original") or {}
        if isinstance(m, dict) and "_error" not in m:
            out[key] = m
    return out


def run_phase_b_one_paper(response_path: str, judge_model: str, run_logger,
                            source_model: str) -> dict | None:
    """Process one paper for Phase B. Loads Script-03 output, judges each filled
    sentence, embeds Phase A judgments, writes per-paper file.

    `source_model` is the CANONICAL config value (e.g., "deepseek-ai/DeepSeek-V3.2"),
    not whatever was written into the file's model field (which may differ
    historically — e.g., legacy OpenRouter slugs like "deepseek/deepseek-v3.2").
    Using the canonical name guarantees one Phase B output dir per source model
    regardless of how the source files were originally written.
    """
    with open(response_path) as f:
        script03 = json.load(f)

    arxiv_id_raw = script03.get("arxiv_id") or os.path.basename(os.path.dirname(response_path))
    arxiv_id = arxiv_folder_name(arxiv_id_raw)
    paper_title = script03.get("title", "")
    paper_year = script03.get("year")
    paper_venue = script03.get("venue", "")

    raw_results = script03.get("results", []) or []
    if not raw_results:
        return None

    # Load Phase A judgments for this paper (keyed by sentence sig)
    phase_a = load_phase_a_motivations(judge_model, arxiv_id)

    out_path = phase_b_paper_path(source_model, judge_model, arxiv_id)
    # Preserve existing successful filled judgments to support resume.
    existing = {}  # context_index -> motivation_judge_filled dict
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                prev = json.load(f)
            for r in prev.get("results", []):
                m = r.get("motivation_judge_filled") or {}
                if isinstance(m, dict) and "_error" not in m and m.get("motivation"):
                    existing[r.get("context_index")] = m
        except Exception:
            pass

    def _judge_one(i_r):
        i, r = i_r
        if i in existing:
            return i, existing[i], r
        # Skip if Script-03 didn't produce a filled sentence (errored at script 03)
        s03_resp = r.get("llm_response", {}) or {}
        if "_error" in s03_resp:
            return i, {"_error": "no_script03_response", "_error_detail": s03_resp.get("_error")}, r
        filled = s03_resp.get("citation_sentence", "") or ""
        if not filled:
            return i, {"_error": "missing_filled_sentence"}, r

        result = classify_with_retry(
            clients=CLIENTS, idx=i, judge_model=judge_model,
            paper_title=paper_title, section=r.get("section", ""),
            citation_sentence=filled,
            before_citation=r.get("before_citation", ""),
            after_citation=r.get("after_citation", ""),
        )
        return i, result, r

    by_idx = {}
    workers = max(1, min(MAX_WORKERS, len(raw_results)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_judge_one, (i, r)) for i, r in enumerate(raw_results)]
        for fut in as_completed(futures):
            i, m, r = fut.result()
            by_idx[i] = (m, r)

    results = []
    for i in sorted(by_idx):
        m_filled, r = by_idx[i]
        s03_resp = r.get("llm_response", {}) or {}
        # Pull motivation_self from Script-03 output
        motivation_self = (
            {
                "motivation": s03_resp.get("motivation"),
                "confidence": s03_resp.get("confidence"),
                "motivation_explanation": s03_resp.get("motivation_explanation"),
            }
            if isinstance(s03_resp, dict) and "_error" not in s03_resp
            else {"_error": s03_resp.get("_error", "no_script03_response")}
        )
        # Look up Phase A original judgment (by section + original sentence)
        key = (r.get("section",""), r.get("citation_sentence_original",""))
        motivation_judge_original = phase_a.get(key, {"_error": "missing_phase_a"})

        results.append({
            "context_index": r.get("context_index", i),
            "source_context_indices": r.get("source_context_indices", []),
            "section": r.get("section", ""),
            "cite_keys": r.get("cite_keys", []),
            "cite_commands": r.get("cite_commands", []),
            "bib_entries": r.get("bib_entries", {}),
            "citation_sentence_original": r.get("citation_sentence_original", ""),
            "citation_sentence_filled": (s03_resp.get("citation_sentence", "") if isinstance(s03_resp, dict) and "_error" not in s03_resp else ""),
            "before_citation": r.get("before_citation", ""),
            "after_citation": r.get("after_citation", ""),
            "masked_paragraph": r.get("masked_paragraph", ""),
            "num_citations_required": r.get("num_citations_required", max(1, len(r.get("cite_keys", [])))),
            "num_citations_returned": r.get("num_citations_returned", 0),
            "recommended_papers": (s03_resp.get("recommended_papers", []) if isinstance(s03_resp, dict) and "_error" not in s03_resp else []),
            "motivation_self": motivation_self,
            "motivation_judge_original": motivation_judge_original,
            "motivation_judge_filled": m_filled,
        })

    out_data = {
        "arxiv_id": script03.get("arxiv_id"),
        "doi": script03.get("doi"),
        "title": paper_title,
        "year": paper_year,
        "venue": paper_venue,
        "preprint_date": script03.get("preprint_date"),
        "first_arxiv_date": script03.get("first_arxiv_date"),
        "latest_arxiv_date": script03.get("latest_arxiv_date"),
        "source_model": source_model,
        "judge_model": judge_model,
        "num_input_results": len(raw_results),
        "num_contexts": len(results),
        "results": results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, default=str)
    os.replace(tmp, out_path)

    n_err = sum(1 for r in results if "_error" in (r["motivation_judge_filled"] or {}))
    if run_logger:
        run_logger.info(
            f"PHASE_B {sanitize_model_slug(source_model)}/{arxiv_id}: ctx={len(results)} err_filled={n_err}"
        )
    return {"arxiv_id": arxiv_id, "num_contexts": len(results), "num_errors": n_err}


def run_phase_b(source_model: str, judge_model: str, run_logger):
    """Iterate Script-03 output for one source model and judge filled sentences."""
    src_slug = sanitize_model_slug(source_model)
    src_dir = os.path.join(RESPONSES_DIR, src_slug)
    response_files = sorted(glob.glob(os.path.join(src_dir, "*/responses.json")))
    if not response_files:
        print(f"[Phase B] no input found for source model {source_model} at {src_dir}")
        return

    # Filter Script-03 outputs by INCLUDE_VENUES_PHASE_B
    if INCLUDE_VENUES_PHASE_B:
        kept = []
        for fp in response_files:
            try:
                with open(fp) as f:
                    venue = get_venue(json.load(f).get("doi", ""))
            except Exception:
                venue = "Unknown"
            if venue in INCLUDE_VENUES_PHASE_B:
                kept.append(fp)
        response_files = kept

    todo = []
    already = 0
    for fp in response_files:
        arxiv_id = os.path.basename(os.path.dirname(fp))
        if phase_b_paper_done(source_model, judge_model, arxiv_id):
            already += 1
        else:
            todo.append(fp)
    if MAX_PAPERS_PHASE_B:
        todo = todo[:MAX_PAPERS_PHASE_B]

    out_dir = phase_b_dir(source_model, judge_model)
    print(f"\n--- PHASE B — judge FILLED sentences ---")
    print(f"Source model     : {source_model}")
    print(f"Judge model      : {judge_model}")
    print(f"Output dir       : {out_dir}")
    print(f"Total papers     : {len(response_files)} (post venue filter)")
    print(f"Already done     : {already}")
    print(f"To judge         : {len(todo)}")
    if run_logger:
        run_logger.info(
            f"PHASE B start: source={source_model} total={len(response_files)} todo={len(todo)}"
        )
    if not todo:
        return

    started = time.time()
    papers_ok = total_ctx = total_err = 0
    pair_slug = f"source__{sanitize_model_slug(source_model)}__judge__{sanitize_model_slug(judge_model)}"
    write_snapshot(pair_slug, {
        "phase": "B", "source_model": source_model, "judge_model": judge_model,
        "total_papers": len(response_files), "todo": len(todo),
        "papers_done_this_run": 0, "elapsed_seconds": 0, "status": "starting",
    })

    pbar = tqdm(todo, desc=f"Phase B ({sanitize_model_slug(source_model)})", unit="paper")
    for idx, fp in enumerate(pbar):
        try:
            summary = run_phase_b_one_paper(fp, judge_model, run_logger, source_model=source_model)
            if summary:
                papers_ok += 1
                total_ctx += summary["num_contexts"]
                total_err += summary["num_errors"]
        except Exception as e:
            if run_logger:
                run_logger.exception(f"PHASE_B FAIL {fp}: {e}")

        elapsed = time.time() - started
        pbar.set_postfix(ok=papers_ok, ctx=total_ctx, err=total_err)
        if (idx + 1) % 5 == 0 or idx == len(todo) - 1:
            ppm = (idx + 1) / elapsed * 60 if elapsed > 0 else 0
            eta = (len(todo) - idx - 1) / ((idx + 1) / elapsed) if (idx + 1) and elapsed > 0 else None
            write_snapshot(pair_slug, {
                "phase": "B", "source_model": source_model, "judge_model": judge_model,
                "total_papers": len(response_files), "todo": len(todo),
                "papers_done_this_run": idx + 1,
                "papers_ok": papers_ok,
                "total_contexts": total_ctx, "total_errors": total_err,
                "elapsed_seconds": round(elapsed, 1),
                "papers_per_minute": round(ppm, 2),
                "eta_seconds": round(eta, 0) if eta else None,
                "status": "running" if idx + 1 < len(todo) else "done",
            })

    if run_logger:
        run_logger.info(
            f"PHASE B done: source={source_model} papers_ok={papers_ok} ctx={total_ctx} err={total_err}"
        )
    print(f"PHASE B summary [{source_model}]: papers_ok={papers_ok} contexts={total_ctx} errors={total_err}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if not CLIENTS:
        raise RuntimeError(
            f"No API keys found. Set {KEYS_FILE}, {ENV_KEYS_PLURAL}, "
            f"{ENV_KEYS_SINGULAR}, or OPENAI_API_KEY."
        )

    judge_model = os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL).strip() or DEFAULT_JUDGE_MODEL
    source_models = parse_models_env("SOURCE_MODELS", DEFAULT_SOURCE_MODELS)

    cutoff_ts = pd.Timestamp(PREPRINT_DATE_CUTOFF)
    preprint_date_lookup = load_preprint_date_lookup(PREPRINT_DATE_CSV)
    arxiv_date_lookup = load_arxiv_date_lookup(ARXIV_DATES_CSV)

    print("=" * 70)
    print("Step 02: LLM-as-Judge intent classification (two-phase)")
    print(f"Judge model      : {judge_model}")
    print(f"Source models    : {source_models}")
    print(f"Provider         : {PROVIDER_BASE_URL}")
    print(f"API keys loaded  : {len(CLIENTS)}")
    print(f"Phase A scope    : full dataset (year={PUBLICATION_YEAR}, first_arxiv_date>{PREPRINT_DATE_CUTOFF}, all venues)")
    print(f"Phase B scope    : INCLUDE_VENUES={sorted(INCLUDE_VENUES_PHASE_B) if INCLUDE_VENUES_PHASE_B else 'all'}")
    print(f"Output base      : {JUDGE_OUTPUT_DIR}/")
    print(f"Phase env (PHASE): {PHASE}")
    print("=" * 70)

    run_logger, log_path = setup_logger(sanitize_model_slug(judge_model))
    run_logger.info(f"Judge: {judge_model}")
    run_logger.info(f"Source models: {source_models}")
    run_logger.info(f"PHASE env: {PHASE}")

    # ------- Phase A inputs: load all paper metadata from MASKED_DIR -------
    if "A" in PHASE:
        print("\nLoading masked-paper metadata for Phase A...")
        meta_files = glob.glob(os.path.join(MASKED_DIR, "*/metadata.json"))
        all_papers = []
        for mf in meta_files:
            paper_dir = os.path.dirname(mf)
            folder = os.path.basename(paper_dir)
            with open(mf) as f:
                meta = json.load(f)
            if not meta.get("preprint_date"):
                meta["preprint_date"] = preprint_date_lookup.get(folder)
            arxiv_dates = arxiv_date_lookup.get(folder, {})
            if not meta.get("first_arxiv_date"):
                meta["first_arxiv_date"] = arxiv_dates.get("first_arxiv_date")
            if not meta.get("latest_arxiv_date"):
                meta["latest_arxiv_date"] = arxiv_dates.get("latest_arxiv_date")
            meta["paper_dir"] = paper_dir
            meta["folder"] = folder
            all_papers.append(meta)

        df_all = pd.DataFrame(all_papers)
        df_all["first_arxiv_date_parsed"] = pd.to_datetime(df_all.get("first_arxiv_date"), errors="coerce")
        df_valid = df_all[
            (pd.to_numeric(df_all["year"], errors="coerce") == PUBLICATION_YEAR) &
            (df_all["first_arxiv_date_parsed"] > cutoff_ts)
        ].copy()
        valid_papers = df_valid.to_dict("records")
        print(f"Phase A scope    : {len(valid_papers)} papers (after year+date filter; venues unrestricted)")

        run_phase_a(judge_model, valid_papers, run_logger)

    # ------- Phase B: per source model -------
    if "B" in PHASE:
        for source_model in source_models:
            run_phase_b(source_model, judge_model, run_logger)

    print("\nStep 02 done.")
    print(f"Log file        : {log_path}")
    print(f"Phase A output  : {phase_a_dir(judge_model)}/")
    for source_model in source_models:
        print(f"Phase B output  : {phase_b_dir(source_model, judge_model)}/")
# %%
