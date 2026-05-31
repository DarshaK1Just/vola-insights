from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


class Config:
    # ── OpenRouter API key ────────────────────────────────────────────────────
    OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")

    # ── LLM models (OpenRouter free tier) ────────────────────────────────────
    MODEL_PRIMARY   = "google/gemma-4-31b-it:free"
    MODEL_FALLBACK_1 = "openai/gpt-oss-120b:free"
    MODEL_FALLBACK_2 = "meta-llama/llama-3.1-8b-instruct:free"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # ── Retry / timeout ───────────────────────────────────────────────────────
    MAX_RETRIES = 3
    BACKOFF_BASE = 2.0
    TIMEOUT_SECONDS = 60

    # ── Circuit breaker ───────────────────────────────────────────────────────
    CIRCUIT_BREAKER_THRESHOLD = 3
    CIRCUIT_RESET_SECONDS = 120
    CIRCUIT_TTL = 300

    # ── Cache TTLs (seconds) ──────────────────────────────────────────────────
    MAX_QUERY_HISTORY = 10      # few-shot window
    PROFILE_TTL = 3600          # 1 hour
    QUERY_HISTORY_TTL = 86400   # 24 hours
    VIZ_STATE_TTL = 1800        # 30 minutes

    # ── Redis semantic cache ─────────────────────────────────────────────────
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    REDIS_ENABLED: bool = os.environ.get("REDIS_ENABLED", "true").lower() not in ("false", "0", "no")
    # How long to cache a full LLM response for repeated identical queries
    RESPONSE_CACHE_TTL: int = int(os.environ.get("RESPONSE_CACHE_TTL", "3600"))   # 1 hour

    # ── Token / input limits ──────────────────────────────────────────────────
    MAX_PROMPT_CHARS = 2000     # input guardrail hard limit
    MAX_OUTPUT_TOKENS = 1500
    MAX_INPUT_TOKENS = 8000     # budget for assembled context
    FEW_SHOT_HISTORY_N = 5      # past interactions injected as few-shot

    # ── Analysis defaults ─────────────────────────────────────────────────────
    TOP_N_CATEGORIES = 7
    HALLUCINATION_TOLERANCE = 0.05   # 5% tolerance for amount cross-checking

    # ── Paths ─────────────────────────────────────────────────────────────────
    _data_file_env = os.environ.get("DATA_FILE", "").strip()
    # Priority: env var > project root > parent directory
    if _data_file_env:
        DATA_FILE: str = _data_file_env
    elif Path(_PROJECT_ROOT / "assessment_transaction_data.xlsx").exists():
        DATA_FILE: str = str(_PROJECT_ROOT / "assessment_transaction_data.xlsx")
    else:
        DATA_FILE: str = str(_PROJECT_ROOT.parent / "assessment_transaction_data.xlsx")

    CACHE_DIR: str = str(_PROJECT_ROOT / "cache_store")
    OUTPUT_DIR: str = str(_PROJECT_ROOT / "output")
    LOG_DIR: str = str(_PROJECT_ROOT / "logs")

    # ── API server ────────────────────────────────────────────────────────────
    API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.environ.get("API_PORT", "8000"))
    CORS_ORIGINS: list = [
        o.strip()
        for o in os.environ.get("CORS_ORIGINS", "*").split(",")
        if o.strip()
    ]
    API_URL: str = os.environ.get("API_URL", "http://localhost:8000")


# Ensure runtime directories exist
for _dir in (Config.CACHE_DIR, Config.OUTPUT_DIR, Config.LOG_DIR):
    Path(_dir).mkdir(parents=True, exist_ok=True)
