"""
Input guardrail — validates the user prompt BEFORE it reaches the LLM.

Checks (in order):
  1. Cross-user data-access detection  (HARD BLOCK)
  2. Prompt injection / jailbreak detection  (HARD BLOCK)
  3. Financial scope enforcement  (HARD BLOCK with polite redirect)
  4. Input length — TRUNCATES gracefully with a warning flag
"""
import warnings
warnings.filterwarnings("ignore", message="Could not obtain an event loop")
warnings.filterwarnings("ignore", category=UserWarning, module="guardrails")

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

try:
    from guardrails import Guard
except ImportError:
    from src.guardrails.compat import Guard  # type: ignore[no-redef]

from src.guardrails.validators import (
    LLMInjectionValidator,   # AI-powered: LLM + regex pattern fallback
    MaxLengthValidator,
    FINANCIAL_KEYWORDS,
    GREETING_PHRASES,
)
from src.config import Config

logger = logging.getLogger(__name__)


# Patterns that indicate the user is requesting ANOTHER user's data.
CROSS_USER_REQUEST_PATTERNS = [
    r"usr_[a-z0-9_]{2,}",
    r"user_[a-z0-9_]{2,}",
    r"tell\s+me\s+about\s+\w+['']s\s+(spend|transact|data|account|finance)",
    r"show\s+me\s+\w+['']s\s+(spend|transact|data|finance)",
    r"give\s+me\s+\w+['']s\s+(financial|transaction|spending)",
    r"access\s+\w+['']s\s+(account|data|transactions)",
    r"(other|another|different)\s+user",
]


@dataclass
class GuardResult:
    passed: bool
    flags: list = field(default_factory=list)
    blocked_response: Optional[str] = None
    effective_prompt: Optional[str] = None


class InputGuard:
    """Validates user-supplied prompts before they enter the pipeline."""

    def __init__(self):
        # LLMInjectionValidator: Tier-1 regex + Tier-2 OpenRouter LLM classifier
        self._injection_guard = Guard()
        self._injection_guard.use(LLMInjectionValidator(on_fail="exception"))

        self._length_guard = Guard()
        self._length_guard.use(
            MaxLengthValidator(max_chars=Config.MAX_PROMPT_CHARS, on_fail="fix")
        )

        try:
            import guardrails.settings as _gs
            if hasattr(_gs, "disable_async"):
                _gs.disable_async = True
        except Exception:
            pass

    def check(self, prompt: str, user_id: str) -> GuardResult:
        flags = []
        effective_prompt = prompt

        # ── 1. Cross-user check ───────────────────────────────────────────────
        mentioned_ids = re.findall(r"usr_[a-z0-9_]+", prompt, re.IGNORECASE)
        if any(uid.lower() != user_id.lower() for uid in mentioned_ids):
            flags.append("CROSS_USER_REQUEST")
            return GuardResult(
                passed=False, flags=flags,
                blocked_response=(
                    "I can only provide information about your own financial data. "
                    "I'm not able to access or share another user's transactions."
                ),
            )

        for pat in CROSS_USER_REQUEST_PATTERNS[1:]:
            if re.search(pat, prompt, re.IGNORECASE):
                flags.append("CROSS_USER_REQUEST")
                return GuardResult(
                    passed=False, flags=flags,
                    blocked_response=(
                        "I can only provide information about your own financial data. "
                        "I'm not able to access or share another user's transactions."
                    ),
                )

        # ── 2. Prompt injection ───────────────────────────────────────────────
        try:
            result = self._injection_guard.validate(prompt)
            if not result.validation_passed:
                flags.append("PROMPT_INJECTION")
                return GuardResult(
                    passed=False, flags=flags,
                    blocked_response=(
                        "I can only answer questions about your own financial transactions. "
                        "Please ask a relevant financial question."
                    ),
                )
        except Exception as exc:
            flags.append("PROMPT_INJECTION")
            logger.warning("Prompt injection exception for user_id=%s: %s", user_id, exc)
            return GuardResult(
                passed=False, flags=flags,
                blocked_response=(
                    "I can only answer questions about your own financial transactions. "
                    "Please ask a relevant financial question."
                ),
            )

        # ── 3. Off-topic scope check ──────────────────────────────────────────
        pl = prompt.lower()
        has_financial = any(kw in pl for kw in FINANCIAL_KEYWORDS)
        has_greeting  = any(ph in pl for ph in GREETING_PHRASES)
        if not has_financial and not has_greeting:
            flags.append("OFF_TOPIC")
            return GuardResult(
                passed=False, flags=flags,
                blocked_response=(
                    "I'm your personal financial assistant. I can help with questions "
                    "about your spending, income, savings, and transaction history. "
                    "Please ask me something about your finances!"
                ),
            )

        # ── 4. Length check — truncate gracefully ─────────────────────────────
        if len(prompt) > Config.MAX_PROMPT_CHARS:
            flags.append("PROMPT_TOO_LONG")
            effective_prompt = prompt[: Config.MAX_PROMPT_CHARS]
            logger.warning(
                "Prompt truncated from %d to %d chars for user_id=%s",
                len(prompt), Config.MAX_PROMPT_CHARS, user_id,
            )

        return GuardResult(
            passed=True,
            flags=flags,
            blocked_response=None,
            effective_prompt=effective_prompt,
        )
