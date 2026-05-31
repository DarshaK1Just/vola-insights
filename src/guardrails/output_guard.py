"""
Output guardrail — validates the LLM response BEFORE it is returned to the user.

Checks:
  1. Toxicity filter                   (HARD BLOCK)
  2. Cross-user data leak              (HARD BLOCK)
  3. Hallucination / number grounding  (soft flag — uses ACTUAL tool call results)
  4. Confidence gating                 (soft flag — surfaces LLM uncertainty)
"""
import warnings
warnings.filterwarnings("ignore", message="Could not obtain an event loop")
warnings.filterwarnings("ignore", category=UserWarning, module="guardrails")

import logging
from dataclasses import dataclass, field
from typing import Optional

try:
    from guardrails import Guard
except ImportError:
    from src.guardrails.compat import Guard  # type: ignore[no-redef]

from src.guardrails.validators import (
    NumbersGroundedValidator,
    NoCrossUserLeakValidator,
    LLMToxicityValidator,    # AI-powered: LLM + keyword pattern fallback
)
from src.guardrails.input_guard import GuardResult

logger = logging.getLogger(__name__)

_LOW_CONF_PHRASES = [
    "i'm not sure", "i am not sure", "i don't know", "i do not know",
    "i cannot determine", "i am unable to", "i'm unable to",
    "i don't have enough", "i do not have enough", "insufficient data",
    "i cannot confirm", "i'm uncertain", "i am uncertain", "i'm not certain",
    "it's unclear", "cannot be confirmed", "i don't have access",
    "i do not have access", "i lack the data", "approximate",
    "rough estimate", "this is an estimate", "i'm guessing", "i am guessing",
    "might be around", "could be approximately", "possibly around",
    "not enough data", "limited data available",
]


class OutputGuard:
    """Validates LLM-generated responses before returning to the user."""

    def __init__(self):
        # LLMToxicityValidator: Tier-1 keyword patterns + Tier-2 OpenRouter LLM
        self._toxicity = Guard()
        self._toxicity.use(LLMToxicityValidator(on_fail="exception"))

        # Deterministic — regex is the correct tool for user-ID pattern matching
        self._cross_user = Guard()
        self._cross_user.use(NoCrossUserLeakValidator(on_fail="exception"))

        # Math-based — deterministic cross-reference against actual tool results
        self._hallucination = Guard()
        self._hallucination.use(NumbersGroundedValidator(on_fail="noop"))

    def check(
        self,
        response: str,
        data_summary: dict,
        user_id: str,
        all_user_ids: list,
        tool_results: Optional[dict] = None,
    ) -> GuardResult:
        flags = []

        # ── 1. Toxicity ───────────────────────────────────────────────────────
        try:
            r = self._toxicity.validate(response)
            if not r.validation_passed:
                flags.append("TOXIC_OUTPUT")
                return GuardResult(
                    passed=False, flags=flags,
                    blocked_response="Response filtered. Please rephrase your question.",
                )
        except Exception as exc:
            flags.append("TOXIC_OUTPUT")
            logger.warning("Toxicity exception for user_id=%s: %s", user_id, exc)
            return GuardResult(passed=False, flags=flags, blocked_response="Response filtered.")

        # ── 2. Cross-user data leak ───────────────────────────────────────────
        try:
            r = self._cross_user.validate(
                response,
                metadata={"user_id": user_id, "all_user_ids": all_user_ids},
            )
            if not r.validation_passed:
                flags.append("CROSS_USER_LEAK")
                return GuardResult(
                    passed=False, flags=flags,
                    blocked_response="I can only share your own financial data.",
                )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "cross" in exc_str or "user" in exc_str:
                flags.append("CROSS_USER_LEAK")
                return GuardResult(
                    passed=False, flags=flags,
                    blocked_response="I can only share your own financial data.",
                )
            logger.error("Cross-user guard unexpected exc for user_id=%s: %s", user_id, exc)

        # ── 3. HALLUCINATION check — cross-reference numbers against tool results
        try:
            hal_meta = {
                "data_summary": data_summary,
                "tool_results": tool_results or {},
                "flags": [],
            }
            self._hallucination.validate(response, metadata=hal_meta)
            hal_flags = [
                "HALLUCINATION" if f == "POTENTIAL_HALLUCINATION" else f
                for f in hal_meta.get("flags", [])
            ]
            if hal_flags:
                logger.info(
                    "HALLUCINATION guard flagged %d ungrounded claim(s) for user_id=%s",
                    len(hal_flags), user_id,
                )
            flags.extend(hal_flags)
        except Exception as exc:
            logger.debug("Hallucination guard non-fatal exc for user_id=%s: %s", user_id, exc)

        # ── 4. Confidence gating (soft flag) ──────────────────────────────────
        resp_lower = response.lower()
        if any(phrase in resp_lower for phrase in _LOW_CONF_PHRASES):
            flags.append("LOW_CONFIDENCE")
            logger.info("Low-confidence response for user_id=%s", user_id)

        return GuardResult(passed=True, flags=flags, blocked_response=None)
