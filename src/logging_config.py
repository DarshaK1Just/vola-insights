"""Central logging configuration for the Vola financial AI pipeline."""
import logging
import os

# ── Set before any library imports to suppress guardrails network calls ───────
# Without these, guardrails-ai tries to reach guardrails.ai on every startup
# which can cause 30-60 s delays when the network is slow or blocked.
os.environ.setdefault("GUARDRAILS_LOG_LEVEL", "ERROR")   # suppress hub warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # suppress HF tokenizer fork warning


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger and suppress third-party noise.

    Called automatically on module import. Safe to call again with a
    different level (e.g. configure_logging(logging.DEBUG) for debugging).
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Suppress chattier third-party loggers to WARNING
    for name in ("httpx", "httpcore", "uvicorn.access", "guardrails", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Guardrails emits a lot of internal DEBUG/INFO — push to ERROR
    logging.getLogger("guardrails").setLevel(logging.ERROR)


# Apply on import so any file that does `from src.logging_config import ...`
# gets logging configured immediately.
configure_logging()
