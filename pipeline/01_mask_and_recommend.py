#%%
"""
Step 01 — Masking & LLM citation reconstruction
===============================================
For each masked citation context, ask an LLM to:
  1. Recommend a paper that should be cited at [CITE_HERE]
  2. Provide bibliographic details (title, authors, year, venue)
  3. Write a citation sentence to replace [CITE_HERE]
  4. Classify the citation motivation (supporting/contrasting/mentioning)

Works with either the OpenRouter or the SiliconFlow API (both OpenAI-compatible).
Select the provider with the PROVIDER env var (default: openrouter):

    PROVIDER=openrouter  python pipeline/01_mask_and_recommend.py     # default
    PROVIDER=siliconflow python pipeline/01_mask_and_recommend.py

Per provider (see PROVIDERS below) this sets the base URL, the keys file/field,
the env-var names, and the default model. Any field can be overridden via env
(PROVIDER_BASE_URL, KEYS_FILE, OPENROUTER_MODELS / SILICONFLOW_MODELS, ...).
Keys load from the provider's JSON file (api_keys.json with the provider's
field), the *_API_KEYS env (comma-separated), the *_API_KEY env, or OPENAI_API_KEY.

Saves results per-paper under:
    data/arxiv_llm_responses/{model_slug}/{arxiv_id}/responses.json
Resume-safe: skips papers that already have responses.

Progress monitoring:
  - tqdm bar in the foreground.
  - Log file at logs/{LOG_TAG}_run_{model_slug}.log
  - Live progress snapshot at logs/{LOG_TAG}_progress_{model_slug}.json

Usage:
    python pipeline/01_mask_and_recommend.py

Requires:
    pip install openai pandas tqdm
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
PREPRINT_DATE_CSV = "data/preprints_acl_dimensions.csv"
ARXIV_DATES_CSV = "data/arxiv_api_dates.csv"
OUTPUT_BASE_DIR = "data/arxiv_llm_responses"
LOG_DIR = "logs"

# ----- Provider selection -------------------------------------------------- #
# One script, two OpenAI-compatible providers. Pick with PROVIDER env var.
# Each entry can be overridden individually by the env vars referenced below.
PROVIDERS = {
    "openrouter": {
        "base_url":        "https://openrouter.ai/api/v1",
        "keys_file":       "api_keys.json",          # combined key file
        "keys_json_field": "openrouter_api_keys",
        "env_keys_plural": "OPENROUTER_API_KEYS",
        "env_keys_single": "OPENROUTER_API_KEY",
        "models_env":      "OPENROUTER_MODELS",
        "default_models":  "openai/gpt-5.1-chat",
        "log_tag":         "03_or",
    },
    "siliconflow": {
        "base_url":        "https://api.siliconflow.com/v1",
        "keys_file":       "api_keys.json",
        "keys_json_field": "siliconflow_api_keys",
        "env_keys_plural": "SILICONFLOW_API_KEYS",
        "env_keys_single": "SILICONFLOW_API_KEY",
        "models_env":      "SILICONFLOW_MODELS",
        "default_models":  "deepseek-ai/DeepSeek-V3.2",
        "log_tag":         "03_sf",
    },
}

PROVIDER = os.getenv("PROVIDER", "openrouter").strip().lower()
if PROVIDER not in PROVIDERS:
    raise SystemExit(f"Unknown PROVIDER={PROVIDER!r}; choose one of {list(PROVIDERS)}")
_P = PROVIDERS[PROVIDER]

# Each constant falls back to the selected provider's default, but an explicit
# env var always wins (so you can point at a custom gateway / key file / model).
PROVIDER_BASE_URL = os.getenv("PROVIDER_BASE_URL", _P["base_url"])
KEYS_FILE         = os.getenv("KEYS_FILE", _P["keys_file"])
KEYS_JSON_FIELD   = _P["keys_json_field"]
ENV_KEYS_PLURAL   = _P["env_keys_plural"]
ENV_KEYS_SINGULAR = _P["env_keys_single"]
MODELS_ENV        = _P["models_env"]
DEFAULT_MODELS    = _P["default_models"]
LOG_TAG           = _P["log_tag"]
MAX_CONTEXTS_PER_PAPER = 100       # Cap to control cost
MAX_WORKERS = 24                   # Parallel requests per paper
MAX_RETRIES = 5                   # Retry on transient api_error / validation_error
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0        # Cap exponential backoff
MAX_PAPERS = None                 # Set to int for testing (e.g., 5)
PREPRINT_DATE_CUTOFF = "2024-12-31"
PUBLICATION_YEAR = 2025

# Restrict run to specific venues (DOI-based). Set to None or empty to disable.
INCLUDE_VENUES = {"ACL Main", "EMNLP Main", "NAACL Main"}


def get_venue(doi) -> str:
    """Classify a DOI into ACL/EMNLP/NAACL Main vs Findings vs Other."""
    if doi is None:
        return "Unknown"
    if isinstance(doi, float) and pd.isna(doi):
        return "Unknown"
    d = str(doi).strip().lower()
    if not d:
        return "Unknown"
    if "findings-emnlp" in d:
        return "Findings-EMNLP"
    if "findings-acl" in d:
        return "Findings-ACL"
    if "findings-naacl" in d:
        return "Findings-NAACL"
    if "emnlp-main" in d:
        return "EMNLP Main"
    if "naacl-main" in d or "naacl-long" in d or "naacl-short" in d:
        return "NAACL Main"
    if "acl-main" in d or "acl-long" in d or "acl-short" in d:
        return "ACL Main"
    return "Other"


def parse_models() -> list[str]:
    """
    Comma-separated model slugs from the provider's models env var
    (OPENROUTER_MODELS or SILICONFLOW_MODELS); falls back to DEFAULT_MODELS.
    """
    raw = os.getenv(MODELS_ENV, DEFAULT_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def sanitize_model_slug(model_slug: str) -> str:
    """
    Convert model slug to a filesystem-safe directory name.
    """
    sanitized = []
    for ch in model_slug:
        if ch.isalnum() or ch in ("-", "_", "."):
            sanitized.append(ch)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_")


def clean_arxiv_id(arxiv_id) -> str:
    """
    Match folder names produced by Script 02.
    """
    if pd.isna(arxiv_id):
        return ""
    return str(arxiv_id).replace("arXiv:", "").replace("arxiv:", "").strip()


def arxiv_folder_name(arxiv_id) -> str:
    return clean_arxiv_id(arxiv_id).replace("/", "_")


def load_preprint_date_lookup(csv_path: str) -> dict:
    """
    Map arXiv folder name -> preprint_date from the Dimensions CSV.
    This lets Step 01 filter old masked metadata that predates the field.
    """
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    if "arxiv_id" not in df.columns or "preprint_date" not in df.columns:
        return {}

    lookup = {}
    for _, row in df[["arxiv_id", "preprint_date"]].dropna(subset=["arxiv_id"]).iterrows():
        folder = arxiv_folder_name(row["arxiv_id"])
        if folder:
            lookup[folder] = row.get("preprint_date")
    return lookup


def load_arxiv_date_lookup(csv_path: str) -> dict:
    """
    Map arXiv folder name -> arXiv API dates from Script 02 cache.
    """
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    required = {"arxiv_id_clean", "first_arxiv_date", "latest_arxiv_date"}
    if not required.issubset(df.columns):
        return {}

    lookup = {}
    for _, row in df[list(required)].dropna(subset=["arxiv_id_clean"]).iterrows():
        folder = arxiv_folder_name(row["arxiv_id_clean"])
        if folder:
            lookup[folder] = {
                "first_arxiv_date": row.get("first_arxiv_date"),
                "latest_arxiv_date": row.get("latest_arxiv_date"),
            }
    return lookup


def completed_response_has_current_metadata(
    response_path: str,
    cutoff_ts: pd.Timestamp,
    publication_year: int,
) -> bool:
    """
    Treat old outputs with missing/out-of-scope first_arxiv_date as stale so reruns
    can overwrite them with current metadata and filtering assumptions.
    """
    try:
        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    first_arxiv_ts = pd.to_datetime(data.get("first_arxiv_date"), errors="coerce")
    year = pd.to_numeric(data.get("year"), errors="coerce")
    return (
        not pd.isna(first_arxiv_ts)
        and first_arxiv_ts > cutoff_ts
        and not pd.isna(year)
        and int(year) == publication_year
    )

def _load_api_keys() -> list:
    """
    Load API keys from env for key rotation.
    Priority:
      1) KEYS_FILE JSON: {"openrouter_api_keys": ["...", "..."]}
      2) OPENROUTER_API_KEYS (comma-separated)
      3) OPENROUTER_API_KEY
      4) OPENAI_API_KEY (fallback)
    """
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
            file_keys = (
                payload.get(KEYS_JSON_FIELD)
                or payload.get("api_keys")
                or []
            )
            if isinstance(file_keys, list):
                keys = [k.strip() for k in file_keys if isinstance(k, str) and k.strip()]
                if keys:
                    return keys
        except Exception:
            pass

    keys_csv = os.getenv(ENV_KEYS_PLURAL, "").strip()
    if keys_csv:
        return [k.strip() for k in keys_csv.split(",") if k.strip()]

    single_key = os.getenv(ENV_KEYS_SINGULAR, "").strip()
    if single_key:
        return [single_key]

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return [openai_key]

    return []


API_KEYS = _load_api_keys()
CLIENTS = [
    OpenAI(api_key=key, base_url=PROVIDER_BASE_URL)
    for key in API_KEYS
]


# ============================================================
# LOGGING + LIVE PROGRESS SNAPSHOT
# ============================================================

def setup_run_logger(model_slug: str) -> logging.Logger:
    """
    Per-model file logger. Console output stays on tqdm (this logger does NOT
    write to stdout to avoid breaking the progress bar). To monitor live:
        tail -f logs/{LOG_TAG}_run_{model_slug}.log
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{LOG_TAG}_run_{model_slug}.log")
    logger = logging.getLogger(f"script03.{LOG_TAG}.{model_slug}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove handlers from previous runs in the same Python process.
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


def write_progress_snapshot(
    model_slug: str,
    state: dict,
) -> str:
    """
    Atomic-ish JSON dump of current run state. Read from another terminal:
        cat logs/{LOG_TAG}_progress_{model_slug}.json
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    snap_path = os.path.join(LOG_DIR, f"{LOG_TAG}_progress_{model_slug}.json")
    tmp_path = snap_path + ".tmp"
    payload = {**state, "snapshot_time": datetime.now().isoformat(timespec="seconds")}
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, snap_path)
    return snap_path


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You reconstruct removed citations from NLP papers. Given a paragraph where [CITE_HERE] replaces one removed citation-bearing sentence, infer the cited paper(s) and the replacement sentence using the surrounding context.

Return ONLY a valid JSON object, no markdown or text outside the JSON:

{
  "citation_count": <int>,
  "recommended_papers": [
    {"title": "...", "authors": "First, Second, ...", "year": <int>, "venue": "...", "doi": "..." or null}
  ],
  "citation_sentence": "<one sentence replacing [CITE_HERE], including the citation>",
  "motivation": "supporting" | "contrasting" | "mentioning",
  "confidence": "high" | "medium" | "low"
}

Rules:
- recommended_papers length must equal citation_count, which must equal the required count from the user prompt.
- Each item is one distinct work; no duplicates, no combining.
- citation_sentence is one complete sentence including the citation text, not a description of it.
- Prefer real, specific papers. If uncertain, give your best guess and set confidence to "low".

Motivation categories:
- supporting: cited work provides evidence, methods, or findings aligned with the citing paper.
- contrasting: cited work is a competing approach, contradicting finding, or baseline the citing paper improves upon.
- mentioning: cited work is referenced for background, definitions, or general acknowledgment.
"""


# ============================================================
# Paragraph cleaning (strip LaTeX clutter to reduce prompt tokens)
# ============================================================

def clean_paragraph_for_llm(text: str, max_chars: int = 1500) -> str:
    """
    Strip LaTeX clutter from masked paragraphs to reduce prompt tokens
    while preserving semantic context for citation recommendation.
    """
    if not text:
        return text

    # 1. Remove entire table environments (the biggest token win).
    text = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "[TABLE]", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}", "[TABLE]", text, flags=re.DOTALL)

    # 2. Remove figure environments.
    text = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "[FIGURE]", text, flags=re.DOTALL)

    # 3. Remove equation environments (handles starred and unstarred).
    text = re.sub(
        r"\\begin\{(equation|align|gather|eqnarray|multline|displaymath)\*?\}.*?\\end\{\1\*?\}",
        "[EQUATION]",
        text,
        flags=re.DOTALL,
    )

    # 3b. Remove inline display math $$...$$ and \[...\].
    text = re.sub(r"\$\$.*?\$\$", "[MATH]", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", "[MATH]", text, flags=re.DOTALL)

    # 3c. Strip simple inline math $...$ (single-dollar). Replace with placeholder
    # only if it's substantial (>3 chars), otherwise drop it.
    def _inline_math_replace(m):
        content = m.group(1)
        return "[MATH]" if len(content) > 10 else ""
    text = re.sub(r"\$([^$\n]{1,200})\$", _inline_math_replace, text)

    # 4. Strip LaTeX formatting commands but keep the argument.
    text = re.sub(
        r"\\(textbf|textit|emph|texttt|underline|textrm|textsc|sys|lm|data)\{([^{}]*)\}",
        r"\2",
        text,
    )

    # 5. Strip structural commands (keep arguments where useful).
    text = re.sub(
        r"\\(section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^{}]*)\}",
        r"\2",
        text,
    )

    # 6. Remove labels and refs.
    text = re.sub(r"\\(label|ref|cref|eqref|pageref|autoref)\{[^{}]*\}", "", text)

    # 7. Remove footnotes — handle one level of nested braces (e.g., \footnote{\url{...}}).
    # Apply twice to handle simple nesting.
    for _ in range(2):
        text = re.sub(r"\\footnote\{[^{}]*\}", "", text)
    # Greedy fallback for stubborn nested footnotes.
    text = re.sub(r"\\footnote\{.*?\}(?=\s|$|[.,;:])", "", text, flags=re.DOTALL)
    text = re.sub(r"\\footnotemark\b", "", text)

    # 8. Simplify cite commands to [CITE] markers.
    text = re.sub(
        r"\\(cite|citep|citet|citealp|citealt|citeauthor|citeyear|citeyearpar|nocite|fullcite|newcite)\*?\{[^{}]*\}",
        "[CITE]",
        text,
    )

    # 9. Strip URL commands.
    text = re.sub(r"\\url\{[^{}]*\}", "", text)
    text = re.sub(r"\\href\{[^{}]*\}\{[^{}]*\}", "", text)

    # 10. Strip remaining bare LaTeX commands.
    text = re.sub(
        r"\\(setlength|tabcolsep|toprule|midrule|bottomrule|cmidrule|scriptsize|"
        r"footnotesize|small|normalsize|large|Large|LARGE|huge|Huge|newline|"
        r"newpage|hline|noindent|centering|caption|multirow|multicolumn|"
        r"checkmark|ul|item|begin|end|forall|exists|sum|prod|int|frac)\b[^\n]*",
        "",
        text,
    )

    # 11. Strip remaining \command{...} that didn't match above.
    text = re.sub(r"\\[a-zA-Z]+\*?\{[^{}]*\}", "", text)

    # 12. Strip remaining \command tokens without args.
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)

    # 13. Collapse whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # 14. Hard cap (safety net). Try to keep [CITE_HERE] in view.
    if len(text) > max_chars:
        cite_pos = text.find("[CITE_HERE]")
        if cite_pos != -1:
            half = max_chars // 2
            start = max(0, cite_pos - half)
            end = min(len(text), cite_pos + half)
            text = (
                ("..." if start > 0 else "")
                + text[start:end]
                + ("..." if end < len(text) else "")
            )
        else:
            text = text[:max_chars] + "..."

    return text


# ============================================================
# Build user prompt for one citation context
# ============================================================

def build_user_prompt(
    paper_title: str,
    paper_year,
    paper_venue: str,
    section: str,
    masked_paragraph: str,
    required_num_citations: int,
    retry_feedback: str = "",
) -> str:
    """
    Build the user prompt for a single citation context.
    The masked paragraph already includes the local previous/next sentence window.
    LaTeX clutter is stripped here (not in the saved JSON) to reduce prompt tokens.
    """
    venue_year = ", ".join(
        filter(
            None,
            [paper_venue or None, str(paper_year) if paper_year else None],
        )
    )
    venue_line = f"Venue/year: {venue_year}" if venue_year else ""

    cleaned_paragraph = clean_paragraph_for_llm(masked_paragraph)

    parts = [
        f'Paper: "{paper_title}"',
        venue_line,
        f"Section: {section}" if section else "",
        "",
        cleaned_paragraph,
        "",
        f"Required citation count: {required_num_citations}",
    ]
    if retry_feedback:
        parts.append(f"Fix from prior attempt: {retry_feedback}")

    return "\n".join(p for p in parts if p)


def context_signature(ctx: dict) -> tuple:
    """
    Signature used to deduplicate repeated raw contexts that point to
    the same citation sentence. Includes before/after citation windows
    so genuinely distinct surrounding contexts aren't merged.
    """
    return (
        ctx.get("section", ""),
        ctx.get("citation_sentence", ""),
        ctx.get("before_citation", ""),
        ctx.get("after_citation", ""),
        ctx.get("masked_paragraph", ""),
    )


def merge_contexts_by_sentence(contexts: list) -> list:
    """
    Merge raw contexts that share the same sentence-level signature.
    Keeps one unique request per citation sentence and aggregates cite_keys.
    """
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


# ============================================================
# Call LLM API
# ============================================================

def parse_json_object_from_text(raw_text: str) -> dict:
    """
    Parse a JSON object from model output, tolerating wrappers like
    markdown code fences and leading/trailing commentary.
    """
    text = (raw_text or "").strip()
    if not text:
        raise json.JSONDecodeError("empty response", text, 0)

    # Remove markdown code fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the largest JSON object span.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("no JSON object found", text, 0)

    candidate = text[start:end + 1]
    return json.loads(candidate)


def create_chat_completion(client: OpenAI, model: str, messages: list, max_tokens: int):
    """
    Attempt JSON-mode first, then gracefully fallback for models/providers
    that do not fully support response_format.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response, True
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return response, False


def call_llm(
    client: OpenAI,
    model: str,
    paper_title: str,
    paper_year,
    paper_venue: str,
    section: str,
    masked_paragraph: str,
    required_num_citations: int,
    retry_feedback: str = "",
) -> dict:
    """
    Send one citation context to the model and parse the response.
    Returns parsed JSON dict or error dict.
    """
    user_prompt = build_user_prompt(
        paper_title=paper_title,
        paper_year=paper_year,
        paper_venue=paper_venue,
        section=section,
        masked_paragraph=masked_paragraph,
        required_num_citations=required_num_citations,
        retry_feedback=retry_feedback,
    )

    try:
        # Tightened budget for the slim schema:
        #   ~170 tokens for k=1, scaling up for multi-citation cases.
        response, used_json_mode = create_chat_completion(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=min(2500, 200 + 200 * required_num_citations),
        )

        raw_text = response.choices[0].message.content.strip()

        # Parse JSON
        result = parse_json_object_from_text(raw_text)

        # Add API metadata
        result["_model"] = model
        result["_json_mode_enforced"] = used_json_mode
        result["_usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        result["_raw_response"] = raw_text

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


def normalize_and_validate_response(result: dict, required_num_citations: int) -> dict:
    """
    Normalize old/new response schemas and validate required count.
    """
    if "_error" in result:
        return result

    papers = (
        result.get("recommended_papers")
        or result.get("recommended_paper")
        or result.get("papers")
        or result.get("citations")
    )
    if isinstance(papers, dict):
        papers = [papers]
    if not isinstance(papers, list):
        # Backward compatibility: older single-paper schema.
        title = result.get("recommended_title")
        authors = result.get("recommended_authors")
        year = result.get("recommended_year")
        venue = result.get("recommended_venue")
        doi = result.get("recommended_doi")
        if title:
            papers = [{
                "title": title,
                "authors": authors,
                "year": year,
                "venue": venue,
                "doi": doi,
            }]
        else:
            papers = []

    normalized_papers = []
    seen = set()
    for p in papers:
        if isinstance(p, str):
            p = {"title": p}
        if not isinstance(p, dict):
            continue
        item = {
            "title": p.get("title") or p.get("paper_title") or p.get("name") or "",
            "authors": p.get("authors") or p.get("author") or "",
            "year": p.get("year") or p.get("publication_year"),
            "venue": p.get("venue") or p.get("journal") or p.get("conference") or "",
            "doi": p.get("doi"),
        }
        dedup_key = (str(item["title"]).strip().lower(), str(item["year"]))
        if not str(item["title"]).strip():
            continue
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        normalized_papers.append(item)

    result["recommended_papers"] = normalized_papers
    result["citation_count"] = len(normalized_papers)
    result["required_citation_count"] = required_num_citations

    if len(normalized_papers) != required_num_citations:
        return {
            "_error": "validation_error",
            "_error_detail": (
                f"expected_{required_num_citations}_papers_got_{len(normalized_papers)}"
            ),
            "_raw_response": result.get("_raw_response", ""),
            "_partial": result,
        }

    citation_sentence = str(result.get("citation_sentence", "")).strip()
    if not citation_sentence:
        citation_sentence = str(result.get("generated_citation_sentence", "")).strip()
    if not citation_sentence:
        citation_sentence = str(result.get("replacement_sentence", "")).strip()
    result["citation_sentence"] = citation_sentence

    if not citation_sentence:
        return {
            "_error": "validation_error",
            "_error_detail": "missing_citation_sentence",
            "_raw_response": result.get("_raw_response", ""),
            "_partial": result,
        }

    return result


def call_llm_with_retry(
    client: OpenAI,
    model: str,
    paper_title: str,
    paper_year,
    paper_venue: str,
    section: str,
    masked_paragraph: str,
    required_num_citations: int,
) -> dict:
    """
    Retry transient API failures with exponential backoff + jitter.
    """
    retry_feedback = ""
    for attempt in range(MAX_RETRIES):
        result = call_llm(
            client=client,
            model=model,
            paper_title=paper_title,
            paper_year=paper_year,
            paper_venue=paper_venue,
            section=section,
            masked_paragraph=masked_paragraph,
            required_num_citations=required_num_citations,
            retry_feedback=retry_feedback,
        )
        result = normalize_and_validate_response(result, required_num_citations)
        if "_error" not in result:
            return result

        err_type = result.get("_error")

        # Per-error-type policy:
        #  - api_error: transient (rate-limit / timeout / 5xx) -> retry with backoff
        #  - validation_error: model returned wrong shape -> retry with feedback
        #  - json_parse_error: almost always truncation against max_tokens cap; the
        #    model already produced as many tokens as the budget allowed, so retrying
        #    will cost the same and likely fail the same way. Skip immediately.
        #  - anything else: unknown -> skip.
        if err_type == "json_parse_error":
            return result
        if err_type not in {"api_error", "validation_error"}:
            return result

        if attempt == MAX_RETRIES - 1:
            return result

        detail = result.get("_error_detail", err_type or "unknown_error")
        retry_feedback = (
            f"{detail}. Return a valid JSON object with exactly "
            f"{required_num_citations} recommended_papers items."
        )
        backoff = min(MAX_BACKOFF_SECONDS, INITIAL_BACKOFF_SECONDS * (2 ** attempt)) + random.uniform(0, 1.0)
        time.sleep(backoff)

    return {
        "_error": "api_error",
        "_error_detail": "retry_exhausted",
    }


# ============================================================
# Process one paper
# ============================================================

def process_one_paper(
    paper_dir: str,
    output_dir: str,
    model: str,
    preprint_date_lookup: dict,
    arxiv_date_lookup: dict,
) -> dict:
    """
    Process all citation contexts for one paper.
    Returns summary dict.
    """
    arxiv_id = os.path.basename(paper_dir)

    # Load metadata
    meta_path = os.path.join(paper_dir, "metadata.json")
    ctx_path = os.path.join(paper_dir, "contexts.json")

    if not os.path.exists(meta_path) or not os.path.exists(ctx_path):
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    if not meta.get("preprint_date"):
        meta["preprint_date"] = preprint_date_lookup.get(arxiv_id)
    arxiv_dates = arxiv_date_lookup.get(arxiv_id, {})
    if not meta.get("first_arxiv_date"):
        meta["first_arxiv_date"] = arxiv_dates.get("first_arxiv_date")
    if not meta.get("latest_arxiv_date"):
        meta["latest_arxiv_date"] = arxiv_dates.get("latest_arxiv_date")
    with open(ctx_path) as f:
        raw_contexts = json.load(f)

    if not raw_contexts:
        return None

    paper_title = meta.get("title", "")
    paper_year = meta.get("year")
    paper_venue = meta.get("venue", "")
    unique_contexts = merge_contexts_by_sentence(raw_contexts)

    # Cap number of unique sentence contexts per paper.
    if len(unique_contexts) > MAX_CONTEXTS_PER_PAPER:
        unique_contexts = unique_contexts[:MAX_CONTEXTS_PER_PAPER]

    def _run_one_context(i_ctx):
        i, ctx = i_ctx
        client = CLIENTS[i % len(CLIENTS)]
        llm_response = call_llm_with_retry(
            client=client,
            model=model,
            paper_title=paper_title,
            paper_year=paper_year,
            paper_venue=paper_venue,
            section=ctx.get("section", ""),
            masked_paragraph=ctx.get("masked_paragraph", ""),
            required_num_citations=ctx.get("num_required_citations", 1),
        )

        return i, {
            "context_index": i,
            "source_context_indices": ctx.get("source_context_indices", []),
            "section": ctx.get("section", ""),
            "cite_keys": ctx.get("cite_keys", []),
            "cite_commands": ctx.get("cite_commands", []),
            "citation_sentence_original": ctx.get("citation_sentence_original", ""),
            "masked_paragraph": ctx.get("masked_paragraph", ""),
            "before_citation": ctx.get("before_citation", ""),
            "after_citation": ctx.get("after_citation", ""),
            "bib_entries": ctx.get("bib_entries", {}),
            "num_citations_required": ctx.get("num_required_citations", 1),
            "num_citations_returned": (
                len(llm_response.get("recommended_papers", []))
                if isinstance(llm_response, dict) and "_error" not in llm_response
                else 0
            ),
            "llm_response": llm_response,
        }

    # Process contexts in parallel and then restore original order.
    results_by_idx = {}
    workers = max(1, min(MAX_WORKERS, len(unique_contexts)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_run_one_context, (i, ctx))
            for i, ctx in enumerate(unique_contexts)
        ]
        for fut in as_completed(futures):
            i, record = fut.result()
            results_by_idx[i] = record

    results = [results_by_idx[i] for i in sorted(results_by_idx)]

    # Save results
    out_dir = os.path.join(output_dir, arxiv_id)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "responses.json"), "w") as f:
        json.dump({
            "arxiv_id": meta.get("arxiv_id"),
            "doi": meta.get("doi"),
            "title": paper_title,
            "year": meta.get("year"),
            "preprint_date": meta.get("preprint_date"),
            "first_arxiv_date": meta.get("first_arxiv_date"),
            "latest_arxiv_date": meta.get("latest_arxiv_date"),
            "venue": meta.get("venue"),
            "model": model,
            "num_raw_contexts": len(raw_contexts),
            "num_contexts": len(results),
            "results": results,
        }, f, indent=2, default=str)

    return {
        "arxiv_id": arxiv_id,
        "num_contexts": len(results),
        "num_errors": sum(1 for r in results if "_error" in r.get("llm_response", {})),
    }


# ============================================================
# Fix-up phase — retry errored contexts in already-saved files
# ============================================================

def report_errored_contexts_in_dir(output_dir: str, model: str, run_logger=None) -> dict:
    """
    Count-only error report. Walks saved responses.json files and tallies the
    number of contexts whose llm_response carries an `_error` field, broken
    down by error type. Does NOT retry anything — strictly informational.
    """
    files = sorted(glob.glob(os.path.join(output_dir, "*/responses.json")))
    papers_with_errors = 0
    total_errors = 0
    by_type = {}
    per_paper_top = []  # list of (n_err, arxiv_id, by_type_dict)

    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            continue

        paper_types = {}
        for r in data.get("results", []):
            resp = r.get("llm_response", {}) or {}
            if isinstance(resp, dict) and "_error" in resp:
                t = resp.get("_error", "unknown")
                paper_types[t] = paper_types.get(t, 0) + 1
                by_type[t] = by_type.get(t, 0) + 1

        if paper_types:
            papers_with_errors += 1
            n_err = sum(paper_types.values())
            total_errors += n_err
            arxiv_id = os.path.basename(os.path.dirname(fp))
            per_paper_top.append((n_err, arxiv_id, paper_types))

    summary = {
        "papers_scanned": len(files),
        "papers_with_errors": papers_with_errors,
        "total_errored_contexts": total_errors,
        "by_type": by_type,
    }

    if run_logger:
        run_logger.info(f"Error report: {summary}")
        # log per-paper for top 20 most-errored
        per_paper_top.sort(reverse=True)
        for n_err, arxiv_id, types in per_paper_top[:20]:
            run_logger.info(f"  {arxiv_id}: {n_err} errors  {types}")

    return summary


def fix_errored_contexts_in_dir(output_dir: str, model: str, run_logger=None) -> dict:
    """
    Scan saved responses.json files in output_dir for contexts whose
    llm_response carries an `_error` field, retry them using the current
    max_tokens budget, and write files back in place.
    """
    files = sorted(glob.glob(os.path.join(output_dir, "*/responses.json")))

    work = []
    initial_errors = 0
    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            continue
        idxs = [
            i for i, r in enumerate(data.get("results", []))
            if isinstance(r.get("llm_response"), dict) and "_error" in r["llm_response"]
        ]
        if idxs:
            work.append((fp, idxs))
            initial_errors += len(idxs)

    summary = {
        "papers_with_errors": len(work),
        "initial_errored_contexts": initial_errors,
        "papers_touched": 0,
        "contexts_attempted": 0,
        "contexts_fixed": 0,
        "contexts_still_error": 0,
    }

    if not work:
        if run_logger:
            run_logger.info("Fix-up phase: no errored contexts found, skipping.")
        return summary

    if run_logger:
        run_logger.info(
            f"Fix-up phase start: papers_with_errors={len(work)} "
            f"errored_contexts={initial_errors}"
        )

    pbar = tqdm(work, desc=f"Fix-up ({sanitize_model_slug(model)})", unit="paper")
    for fp, idxs in pbar:
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            continue

        paper_title = data.get("title", "")
        paper_year = data.get("year")
        paper_venue = data.get("venue", "")
        arxiv_id = os.path.basename(os.path.dirname(fp))

        def _retry(i):
            r = data["results"][i]
            client = CLIENTS[i % len(CLIENTS)]
            result = call_llm_with_retry(
                client=client,
                model=model,
                paper_title=paper_title,
                paper_year=paper_year,
                paper_venue=paper_venue,
                section=r.get("section", ""),
                masked_paragraph=r.get("masked_paragraph", ""),
                required_num_citations=r.get(
                    "num_citations_required",
                    max(1, len(r.get("cite_keys", []))),
                ),
            )
            return i, result

        n_attempt = len(idxs)
        n_fixed = n_still = 0
        workers = max(1, min(MAX_WORKERS, n_attempt))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_retry, i) for i in idxs]
            for fut in as_completed(futures):
                i, result = fut.result()
                data["results"][i]["llm_response"] = result
                data["results"][i]["num_citations_returned"] = (
                    len(result.get("recommended_papers", []))
                    if isinstance(result, dict) and "_error" not in result
                    else 0
                )
                if "_error" in result:
                    n_still += 1
                else:
                    n_fixed += 1

        data["fix_up_at"] = datetime.now().isoformat(timespec="seconds")
        tmp = fp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, fp)

        summary["papers_touched"] += 1
        summary["contexts_attempted"] += n_attempt
        summary["contexts_fixed"] += n_fixed
        summary["contexts_still_error"] += n_still

        pbar.set_postfix(fixed=summary["contexts_fixed"], still=summary["contexts_still_error"])
        if run_logger:
            run_logger.info(
                f"FIXUP {arxiv_id}: attempted={n_attempt} fixed={n_fixed} still={n_still}"
            )

    if run_logger:
        run_logger.info(f"Fix-up phase done: {summary}")
    return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not CLIENTS:
        raise RuntimeError(
            f"No API keys found. Set {KEYS_FILE}, {ENV_KEYS_PLURAL}, "
            f"{ENV_KEYS_SINGULAR}, or OPENAI_API_KEY."
        )

    models = parse_models()
    cutoff_ts = pd.Timestamp(PREPRINT_DATE_CUTOFF)

    print("=" * 70)
    print(f"LLM Citation Recommendation (masked-citation reconstruction) — {PROVIDER}")
    print(f"Models: {models}")
    print(f"Provider base URL: {PROVIDER_BASE_URL}")
    print(f"Loaded API keys: {len(CLIENTS)}")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Filter: year == {PUBLICATION_YEAR} and first_arxiv_date > {PREPRINT_DATE_CUTOFF}")
    print(f"Base output: {OUTPUT_BASE_DIR}/{{model_slug}}/{{arxiv_id}}/responses.json")
    print(f"Logs:        {LOG_DIR}/03_run_{{model_slug}}.log")
    print(f"Live state:  {LOG_DIR}/03_progress_{{model_slug}}.json")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: Load all paper metadata into a DataFrame
    # --------------------------------------------------------
    print("Loading all metadata...")
    preprint_date_lookup = load_preprint_date_lookup(PREPRINT_DATE_CSV)
    arxiv_date_lookup = load_arxiv_date_lookup(ARXIV_DATES_CSV)
    print(f"Preprint dates loaded from {PREPRINT_DATE_CSV}: {len(preprint_date_lookup)}")
    print(f"arXiv API dates loaded from {ARXIV_DATES_CSV}: {len(arxiv_date_lookup)}")

    meta_files = glob.glob(os.path.join(MASKED_DIR, "*/metadata.json"))
    records = []
    for mf in meta_files:
        paper_dir = os.path.dirname(mf)
        folder = os.path.basename(paper_dir)
        bib_path = os.path.join(paper_dir, "bib_entries.json")
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
        meta["has_bib"] = os.path.exists(bib_path) and os.path.getsize(bib_path) > 10
        records.append(meta)

    df_all = pd.DataFrame(records)
    print(f"Total papers in {MASKED_DIR}: {len(df_all)}")

    # --------------------------------------------------------
    # Step 2: Filter papers by preprint_date and valid bib
    # --------------------------------------------------------
    df_all["first_arxiv_date_parsed"] = pd.to_datetime(
        df_all.get("first_arxiv_date"),
        errors="coerce",
    )
    df_all["venue_short"] = df_all.get("doi", pd.Series([""] * len(df_all))).map(get_venue)
    df_valid = df_all[
        (df_all["has_bib"]) &
        (pd.to_numeric(df_all["year"], errors="coerce") == PUBLICATION_YEAR) &
        (df_all["first_arxiv_date_parsed"] > cutoff_ts)
    ].copy()
    print(
        f"Papers with .bib, year == {PUBLICATION_YEAR}, "
        f"and first_arxiv_date > {PREPRINT_DATE_CUTOFF}: {len(df_valid)}"
    )

    print("Venue distribution before venue filter:")
    for venue, count in df_valid["venue_short"].value_counts().items():
        print(f"  {venue:18s} {count:5d}")

    if INCLUDE_VENUES:
        df_valid = df_valid[df_valid["venue_short"].isin(INCLUDE_VENUES)].copy()
        print(f"After venue filter (kept {sorted(INCLUDE_VENUES)}): {len(df_valid)}")

    missing_first_arxiv = df_all["first_arxiv_date_parsed"].isna().sum()
    if missing_first_arxiv:
        print(f"Papers missing/invalid first_arxiv_date: {missing_first_arxiv}")

    # --------------------------------------------------------
    # Step 3: Run per model with model-specific resume/output
    # --------------------------------------------------------
    for model in models:
        model_dir = sanitize_model_slug(model)
        output_dir = os.path.join(OUTPUT_BASE_DIR, model_dir)
        os.makedirs(output_dir, exist_ok=True)

        run_logger, log_path = setup_run_logger(model_dir)
        run_logger.info(f"Model: {model}")
        run_logger.info(f"Provider: {PROVIDER_BASE_URL}")
        run_logger.info(f"API keys loaded: {len(CLIENTS)}")
        run_logger.info(f"Max workers: {MAX_WORKERS}")
        run_logger.info(f"Output dir: {output_dir}")

        # Phase 1: per-context fix-up of errored contexts in already-saved files.
        # Only failing contexts are retried; their results are written back to
        # the same position in responses.json. Successful contexts are untouched.
        print(f"\nFix-up phase for {model_dir}: scanning {output_dir} for errored contexts...")
        fix_summary = fix_errored_contexts_in_dir(output_dir, model, run_logger)
        print(f"  Papers with errors:      {fix_summary['papers_with_errors']}")
        print(f"  Initial errored ctxs:    {fix_summary['initial_errored_contexts']}")
        print(f"  Contexts fixed:          {fix_summary['contexts_fixed']}")
        print(f"  Contexts still error:    {fix_summary['contexts_still_error']}")

        already_done = {
            d for d in os.listdir(output_dir)
            if completed_response_has_current_metadata(
                os.path.join(output_dir, d, "responses.json"),
                cutoff_ts,
                PUBLICATION_YEAR,
            )
        }

        df_todo = df_valid[~df_valid["folder"].isin(already_done)]
        to_process = df_todo["paper_dir"].tolist()

        if MAX_PAPERS:
            to_process = to_process[:MAX_PAPERS]

        print(f"\n{'-' * 70}")
        print(f"Model: {model}")
        print(f"Output dir: {output_dir}")
        print(f"Log file:   {log_path}")
        print(f"Already processed: {len(already_done)}")
        print(f"To process: {len(to_process)}")
        if MAX_PAPERS:
            print(f"  (capped to {MAX_PAPERS} papers for testing)")
        run_logger.info(f"Already processed: {len(already_done)}")
        run_logger.info(f"To process: {len(to_process)}")

        # Cost estimate (rough; OpenRouter pricing varies by model).
        est_calls = len(to_process) * 50
        avg_input_tokens_per_call = 500
        avg_output_tokens_per_call = 220
        est_input_tokens = est_calls * avg_input_tokens_per_call
        est_output_tokens = est_calls * avg_output_tokens_per_call

        print(f"  Estimated API calls:    ~{est_calls:,}")
        print(f"  Estimated input tokens: ~{est_input_tokens:,}")
        print(f"  Estimated output tokens:~{est_output_tokens:,}")

        papers_ok = 0
        total_contexts = 0
        total_errors = 0
        run_started = time.time()
        total_papers_to_process = len(to_process)

        # Initial snapshot so the user can read it even before paper #1 finishes.
        write_progress_snapshot(model_dir, {
            "model": model,
            "provider_base_url": PROVIDER_BASE_URL,
            "output_dir": output_dir,
            "log_file": log_path,
            "already_processed": len(already_done),
            "to_process": total_papers_to_process,
            "papers_done_this_run": 0,
            "papers_remaining": total_papers_to_process,
            "total_contexts_sent": 0,
            "total_api_errors": 0,
            "current_paper": None,
            "elapsed_seconds": 0,
            "papers_per_minute": 0.0,
            "eta_seconds": None,
            "status": "starting",
        })

        pbar = tqdm(to_process, desc=f"Papers ({model_dir})", unit="paper")
        for idx, paper_dir in enumerate(pbar):
            current_paper = os.path.basename(paper_dir)
            try:
                summary = process_one_paper(
                    paper_dir=paper_dir,
                    output_dir=output_dir,
                    model=model,
                    preprint_date_lookup=preprint_date_lookup,
                    arxiv_date_lookup=arxiv_date_lookup,
                )
                if summary:
                    papers_ok += 1
                    total_contexts += summary["num_contexts"]
                    total_errors += summary["num_errors"]
                    run_logger.info(
                        f"OK [{idx + 1}/{total_papers_to_process}] {current_paper} "
                        f"contexts={summary['num_contexts']} errors={summary['num_errors']}"
                    )
                else:
                    run_logger.warning(
                        f"SKIP [{idx + 1}/{total_papers_to_process}] {current_paper} "
                        f"(no metadata/contexts)"
                    )
            except Exception as e:
                run_logger.exception(
                    f"FAIL [{idx + 1}/{total_papers_to_process}] {current_paper}: {e}"
                )

            elapsed = time.time() - run_started
            done_this_run = idx + 1
            remaining = total_papers_to_process - done_this_run
            ppm = (done_this_run / elapsed * 60) if elapsed > 0 else 0.0
            eta_sec = (remaining / (done_this_run / elapsed)) if done_this_run and elapsed > 0 else None

            pbar.set_postfix(
                ok=papers_ok,
                contexts=total_contexts,
                errors=total_errors,
                ppm=f"{ppm:.1f}",
            )

            write_progress_snapshot(model_dir, {
                "model": model,
                "provider_base_url": PROVIDER_BASE_URL,
                "output_dir": output_dir,
                "log_file": log_path,
                "already_processed": len(already_done),
                "to_process": total_papers_to_process,
                "papers_done_this_run": done_this_run,
                "papers_remaining": remaining,
                "papers_ok": papers_ok,
                "total_contexts_sent": total_contexts,
                "total_api_errors": total_errors,
                "current_paper": current_paper,
                "elapsed_seconds": round(elapsed, 1),
                "papers_per_minute": round(ppm, 2),
                "eta_seconds": round(eta_sec, 0) if eta_sec else None,
                "status": "running" if remaining > 0 else "finishing",
            })

        write_progress_snapshot(model_dir, {
            "model": model,
            "provider_base_url": PROVIDER_BASE_URL,
            "output_dir": output_dir,
            "log_file": log_path,
            "already_processed": len(already_done),
            "to_process": total_papers_to_process,
            "papers_done_this_run": total_papers_to_process,
            "papers_remaining": 0,
            "papers_ok": papers_ok,
            "total_contexts_sent": total_contexts,
            "total_api_errors": total_errors,
            "current_paper": None,
            "elapsed_seconds": round(time.time() - run_started, 1),
            "papers_per_minute": round((total_papers_to_process / max(time.time() - run_started, 1e-9)) * 60, 2),
            "eta_seconds": 0,
            "status": "done",
        })
        run_logger.info(
            f"Run done: papers_ok={papers_ok} contexts={total_contexts} errors={total_errors}"
        )

        print(f"\nSummary for model: {model}")
        print(f"  Papers processed:    {papers_ok}")
        print(f"  Total contexts sent: {total_contexts}")
        print(f"  API errors:          {total_errors}")
        if papers_ok:
            print(f"  Avg contexts/paper:  {total_contexts / papers_ok:.1f}")
        print(f"  Output: {output_dir}/")
        print(f"  Log:    {log_path}")
# %% 
