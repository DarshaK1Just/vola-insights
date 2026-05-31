import hashlib
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class AuditLogger:
    """Append-only JSONL audit logger for all RAG pipeline requests.

    Each log entry records metadata about a request without storing the raw
    prompt text — only a truncated SHA-256 hash is persisted for privacy.
    """

    def __init__(self, log_dir: str) -> None:
        """Initialise the audit logger.

        Args:
            log_dir: Directory where audit.jsonl will be written.
                     Created automatically if it does not exist.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.audit_file = os.path.join(log_dir, "audit.jsonl")
        logger.info("AuditLogger initialised. Writing to %s", self.audit_file)

    def log_request(
        self,
        user_id: str,
        prompt: str,
        response_length: int,
        latency_ms: float,
        cache_hit: bool,
        guardrail_flags: list,
        model_used: str,
    ) -> None:
        """Append one audit record to the JSONL file.

        Args:
            user_id: Identifier of the requesting user.
            prompt: Raw prompt text — only a 16-char hash is stored.
            response_length: Character length of the generated response.
            latency_ms: End-to-end request latency in milliseconds.
            cache_hit: True if the response was served from cache.
            guardrail_flags: List of guardrail trigger strings (may be empty).
            model_used: Model identifier that produced the response.
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "prompt_hash": prompt_hash,
            "response_length": response_length,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "guardrail_flags": guardrail_flags,
            "model_used": model_used,
        }

        try:
            with open(self.audit_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            logger.debug("Audit record written for user %s (prompt_hash=%s)", user_id, prompt_hash)
        except OSError as exc:
            logger.error("Failed to write audit record: %s", exc)
