"""
src/llm/summarizer.py
=====================
Generates mini-summaries (queries) for each ementa in the corpus using the
vLLM Manager endpoint.

Each ementa produces one query saved to data/corpus/queries.json with the
following structure:

    {
        "numero_processo": "...",
        "cdacordao":       "...",
        "classe":          "...",
        "query":           "...",   # generated mini-summary (≤ 300 words)
        "word_count":      ...      # word count of the original ementa
    }

Usage:
    python -m src.llm.summarizer
"""

import json
import os
import time
import logging
import sys
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from src.assets import (
    VLLM_BASE_URL,
    DEFAULT_MODEL,
    LLM_AGENT_TIMEOUT,
    LLM_TEMPERATURE,
    SUMMARY_MAX_TOKENS,
    SUMMARY_MAX_WORDS,
    PROMPT_MINI_RESUMO,
    CORPUS_PATH,
    QUERIES_PATH,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()
VLLM_TOKEN = os.getenv("VLLM_TOKEN")

# ── OpenAI-compatible client pointing to vLLM Manager ─────────────────────────
client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_TOKEN,
    timeout=LLM_AGENT_TIMEOUT,
)


def load_prompt_template(path: str) -> str:
    """Loads the prompt template from disk."""
    return Path(path).read_text(encoding="utf-8")


def build_prompt(template: str, classe: str, texto: str) -> str:
    """Fills the prompt template with ementa data."""
    return template.replace("{classe}", classe).replace("{texto}", texto)


def generate_mini_summary(prompt: str) -> str:
    """
    Calls the vLLM Manager and returns the generated mini-summary.
    Truncates to SUMMARY_MAX_CHARS if the model exceeds the limit.
    """
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=SUMMARY_MAX_TOKENS,
    )
    summary = response.choices[0].message.content.strip()

    # Hard cap: truncate to SUMMARY_MAX_WORDS if the model exceeds the limit
    words = summary.split()
    if len(words) > SUMMARY_MAX_WORDS:
        summary = " ".join(words[:SUMMARY_MAX_WORDS])
        log.warning("Summary truncated to %d words.", SUMMARY_MAX_WORDS)

    return summary


def load_corpus(path: str) -> list[dict]:
    """Loads the ementa corpus from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_existing_queries(path: str) -> dict[str, dict]:
    """
    Loads already-generated queries (if the output file exists) and returns
    them indexed by cdacordao, enabling resume-on-failure.
    """
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["cdacordao"]: item for item in data}


def save_queries(queries: list[dict], path: str) -> None:
    """Saves the queries list to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)


def run(delay_between_requests: float = 0.5) -> None:
    """
    Main pipeline:
      1. Load corpus and prompt template.
      2. For each ementa, generate a mini-summary via the vLLM Manager.
      3. Save results incrementally to QUERIES_PATH (resume-safe).

    Args:
        delay_between_requests: seconds to wait between API calls to avoid
                                 overloading the server.
    """
    log.info("Loading corpus from '%s' ...", CORPUS_PATH)
    corpus = load_corpus(CORPUS_PATH)
    log.info("%d ementas loaded.", len(corpus))

    template = load_prompt_template(PROMPT_MINI_RESUMO)
    log.info("Prompt template loaded from '%s'.", PROMPT_MINI_RESUMO)

    # Resume support: skip ementas already processed
    existing = load_existing_queries(QUERIES_PATH)
    log.info("%d queries already generated (will skip).", len(existing))

    queries: list[dict] = list(existing.values())
    processed = 0
    failed = 0

    """==== Gerar APENAS 1 Query: ===="""
    # ementa = corpus[0]
    # cdacordao = ementa["cdacordao"]

    # log.info("[%d/%d] Generating query for %s ...", 1, len(corpus), ementa["numero_processo"])

    # try:
    #     prompt  = build_prompt(template, ementa["classe"], ementa["texto"])
    #     summary = generate_mini_summary(prompt)

    #     queries.append({
    #         "numero_processo": ementa["numero_processo"],
    #         "cdacordao":       cdacordao,
    #         "classe":          ementa["classe"],
    #         "query":           summary,
    #         "word_count":      ementa.get("word_count", 0),
    #     })

    #     log.info("    ✓ %d words → '%s'", len(summary.split()), summary[:80])
    #     processed += 1

    # except Exception as exc:
    #     log.error("    ✗ Failed for %s: %s", ementa["numero_processo"], exc)
    #     failed += 1

    # save_queries(queries, QUERIES_PATH)

    for i, ementa in enumerate(corpus, start=1):
        cdacordao = ementa["cdacordao"]

        if cdacordao in existing:
            continue  # already done

        log.info("[%d/%d] Generating query for %s ...", i, len(corpus), ementa["numero_processo"])

        try:
            prompt  = build_prompt(template, ementa["classe"], ementa["texto"])
            summary = generate_mini_summary(prompt)

            queries.append({
                "numero_processo": ementa["numero_processo"],
                "cdacordao":       cdacordao,
                "classe":          ementa["classe"],
                "query":           summary,
                "word_count":      ementa.get("word_count", 0),
            })

            log.info("    ✓ %d words → '%s'", len(summary.split()), summary[:80])
            processed += 1

        except Exception as exc:
            log.error("    ✗ Failed for %s: %s", ementa["numero_processo"], exc)
            failed += 1

        # Save incrementally after every ementa (resume-safe)
        save_queries(queries, QUERIES_PATH)

        time.sleep(delay_between_requests)

    log.info("=" * 60)
    log.info("Done. Generated: %d | Skipped: %d | Failed: %d", processed, len(existing), failed)
    log.info("Queries saved to '%s'.", QUERIES_PATH)


if __name__ == "__main__":
    run()