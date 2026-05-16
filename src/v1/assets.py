# ── vLLM Manager ──────────────────────────────────────────────────────────────
VLLM_BASE_URL = "https://llm.liaufms.org/v1/gemma-3-12b-it"
DEFAULT_MODEL  = "google/gemma-3-12b-it"
LLM_AGENT_TIMEOUT = 120

# ── Generation parameters ─────────────────────────────────────────────────────
LLM_TEMPERATURE  = 0.7
SUMMARY_MAX_TOKENS   = 450   # 300 words in Portuguese ~ 400-450 tokens
SUMMARY_MAX_WORDS    = 300   # hard cap enforced after generation

# ── Paths ─────────────────────────────────────────────────────────────────────
PROMPT_MINI_RESUMO = "src/llm/prompts/prompt_geracao_mini_resumo.txt"
CORPUS_PATH        = "data/corpus/ementas.json"
QUERIES_PATH       = "data/queries/queries.json"