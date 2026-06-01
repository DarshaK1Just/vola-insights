"""
Redis-backed semantic KV cache for the financial AI pipeline.

TWO layers:

  Layer 1 — Semantic response cache (NEW)
  ─────────────────────────────────────────
  Key:   resp:{sha256(user_id + normalised_prompt)}
  Value: full pipeline response dict (JSON)
  TTL:   RESPONSE_CACHE_TTL (default 1 h)

  On repeated / near-identical queries the entire LLM pipeline is bypassed
  and the response is served in < 1 ms instead of 10-60 s.

  "Near-identical" means:
    • Case-insensitive  ("FOOD" == "food")
    • Punctuation-stripped  ("last month?" == "last month")
    • Whitespace-normalised  ("last  month" == "last month")

  Layer 2 — User state (mirrors diskcache but in-memory)
  ──────────────────────────────────────────────────────
  Key:   user:{id}:profile         (assessment spec)
  Key:   user:{id}:query_history   (assessment spec)
  Key:   user:{id}:viz_state       (assessment spec)

  Redis reads are ~0.1 ms vs ~2 ms for diskcache.  The diskcache
  UserCacheManager is kept as the fallback (and for circuit-breaker state).

Fallback behaviour
──────────────────
If the redis package is not installed, or if the Redis server is not
reachable, every method falls back silently to the supplied
`fallback_cache` (a UserCacheManager backed by diskcache).  The pipeline
never fails because of Redis.

Setup
─────
  # Option A — local Redis
  redis-server --daemonize yes

  # Option B — Docker (fastest)
  docker run -d --name vola-redis -p 6379:6379 redis:7-alpine

  # Set URL in .env
  REDIS_URL=redis://localhost:6379/0
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Optional redis import ─────────────────────────────────────────────────────
try:
    import redis as _redis_lib           # pip install redis>=5.0
    _REDIS_PKG_AVAILABLE = True
except ImportError:
    _REDIS_PKG_AVAILABLE = False
    logger.warning(
        "redis package not installed (pip install redis). "
        "Semantic response cache disabled — diskcache fallback active."
    )


# ── Prompt normalisation ──────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """
    Normalise a prompt string before hashing for the semantic cache.

    Catches common query variations so they share the same cache entry:
      • "What did I spend?" == "what did i spend?" (case)
      • "last month??"     == "last month"          (punctuation)
      • "last  month"      == "last month"           (whitespace)
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)   # replace all non-word chars with space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _response_key(user_id: str, prompt: str) -> str:
    """Return the Redis key for a (user_id, prompt) response cache entry."""
    normalised = _normalise(prompt)
    digest = hashlib.sha256(f"{user_id}::{normalised}".encode()).hexdigest()
    return f"resp:{digest}"


def _cacheable(response: dict) -> bool:
    """Return True only when a pipeline response is safe to cache."""
    if response.get("error"):
        return False
    blocked_flags = {
        "PROMPT_INJECTION", "CROSS_USER_REQUEST", "OFF_TOPIC",
        "CIRCUIT_OPEN", "PIPELINE_ERROR", "LLM_UNAVAILABLE",
    }
    flags = set(response.get("guardrail_flags", []))
    if flags & blocked_flags:
        return False
    if not response.get("response"):
        return False
    return True


# ── Main class ────────────────────────────────────────────────────────────────

class RedisSemanticCache:
    """
    Redis-backed semantic KV cache with transparent diskcache fallback.

    Parameters
    ----------
    redis_url      : Redis connection URL (default redis://localhost:6379/0).
                     Override via REDIS_URL env var or pass directly.
    response_ttl   : Seconds to keep a cached LLM response (default 3600).
    profile_ttl    : Seconds to keep a user profile (default 3600).
    history_ttl    : Seconds to keep query history (default 86400).
    viz_ttl        : Seconds to keep viz state (default 1800).
    fallback_cache : UserCacheManager instance used when Redis is unavailable.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        response_ttl: int = 3600,
        profile_ttl: int = 3600,
        history_ttl: int = 86400,
        viz_ttl: int = 1800,
        max_history: int = 10,
        fallback_cache=None,
    ):
        self._response_ttl = response_ttl
        self._profile_ttl = profile_ttl
        self._history_ttl = history_ttl
        self._viz_ttl = viz_ttl
        self._max_history = max_history
        self._fallback = fallback_cache
        self._client = None
        self._available = False

        if not _REDIS_PKG_AVAILABLE:
            return

        try:
            # protocol=2 → force RESP2 and skip the HELLO command.
            # The Windows Redis build (3.x) predates HELLO (added in Redis 6.0).
            # redis-py v5 sends HELLO 3 by default; that fails on Redis < 6.
            client = _redis_lib.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                protocol=2,
            )
            client.ping()                      # ~1 ms connectivity check
            self._client = client
            self._available = True
            logger.info("Redis connected at %s — semantic response cache enabled", redis_url)
        except Exception as exc:
            logger.warning(
                "Redis not reachable at %s (%s). "
                "Falling back to diskcache — pipeline unaffected.",
                redis_url, exc,
            )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True when Redis is reachable and the semantic cache is active."""
        return self._available

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rget(self, key: str) -> Optional[dict]:
        """Safe Redis GET → parsed dict, or None on any error."""
        try:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.debug("Redis GET '%s' failed: %s", key, exc)
            return None

    def _rsetex(self, key: str, ttl: int, value: dict) -> bool:
        """Safe Redis SETEX with JSON serialisation. Returns True on success."""
        try:
            self._client.setex(key, ttl, json.dumps(value, default=str))
            return True
        except Exception as exc:
            logger.debug("Redis SETEX '%s' failed: %s", key, exc)
            return False

    # ── Layer 1: Semantic response cache ──────────────────────────────────────

    def get_response(self, user_id: str, prompt: str) -> Optional[dict]:
        """
        Return a cached pipeline response for (user_id, prompt), or None.

        Normalises the prompt before hashing so minor variations (case,
        punctuation, extra spaces) share the same cache entry.
        
        Note: Cached responses exclude chart paths since chart files are ephemeral.
        """
        if not self._available:
            return None
        key = _response_key(user_id, prompt)
        data = self._rget(key)
        if data:
            logger.info(
                "Semantic cache HIT for user=%s prompt='%s...' key=%s",
                user_id, prompt[:40], key[:16],
            )
            # Mark this response as a cache hit
            data["cache_hit"] = True
            # Ensure visualizations is an empty list (charts not cached)
            data.setdefault("visualizations", [])
            return data
        return None

    def set_response(self, user_id: str, prompt: str, response: dict) -> None:
        """Cache a pipeline response with tool calls for chart regeneration."""
        if not self._available:
            return
        if not _cacheable(response):
            logger.debug("Skipping cache for blocked/error response (user=%s)", user_id)
            return
        key = _response_key(user_id, prompt)
        
        # Strip non-JSON-serialisable fields before caching
        safe = {
            k: v for k, v in response.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
        
        # Store tool calls for chart regeneration (keep only viz tool calls)
        if "tool_calls" in safe:
            from src.visualizations import VIZ_TOOL_NAMES
            viz_tool_calls = [
                {"name": tc["name"], "args": tc.get("args", {})}
                for tc in safe["tool_calls"]
                if tc.get("name") in VIZ_TOOL_NAMES
            ]
            safe["cached_tool_calls"] = viz_tool_calls
            # Remove full tool_calls (contains results, too large)
            safe.pop("tool_calls", None)
        
        # Remove chart paths (will be regenerated on cache hit)
        safe.pop("visualizations", None)
        
        if self._rsetex(key, self._response_ttl, safe):
            logger.debug(
                "Semantic cache SET user=%s prompt='%s...' ttl=%ds (%d viz tools cached)",
                user_id, prompt[:40], self._response_ttl, len(safe.get("cached_tool_calls", [])),
            )

    def invalidate_user_responses(self, user_id: str) -> int:
        """
        Delete all cached responses for a user (call when their data changes).

        Uses Redis SCAN — non-blocking even on large keyspaces.
        Returns the number of keys deleted.
        """
        if not self._available:
            return 0
        deleted = 0
        try:
            # We cannot scope by user_id in the key (hashed), so scan all resp: keys
            for key in self._client.scan_iter("resp:*", count=200):
                self._client.delete(key)
                deleted += 1
        except Exception as exc:
            logger.warning("Redis invalidate_user_responses error: %s", exc)
        return deleted

    def clear_all_caches(self) -> dict:
        """
        Clear ALL Redis cache entries (responses, profiles, history, viz state).
        
        Returns a dict with counts of deleted keys by type.
        Use with caution - this is a full cache flush.
        """
        if not self._available:
            return {"error": "Redis not available"}
        
        result = {
            "responses": 0,
            "profiles": 0,
            "history": 0,
            "viz_state": 0,
            "total": 0
        }
        
        try:
            # Clear response cache
            for key in self._client.scan_iter("resp:*", count=500):
                self._client.delete(key)
                result["responses"] += 1
            
            # Clear user profiles
            for key in self._client.scan_iter("user:*:profile", count=500):
                self._client.delete(key)
                result["profiles"] += 1
            
            # Clear query history
            for key in self._client.scan_iter("user:*:query_history", count=500):
                self._client.delete(key)
                result["history"] += 1
            
            # Clear viz state
            for key in self._client.scan_iter("user:*:viz_state", count=500):
                self._client.delete(key)
                result["viz_state"] += 1
            
            result["total"] = sum(v for k, v in result.items() if k != "total")
            logger.info("Cleared all Redis caches: %s", result)
            return result
            
        except Exception as exc:
            logger.error("clear_all_caches failed: %s", exc)
            return {"error": str(exc)}

    def cache_stats(self) -> dict:
        """Return basic cache statistics (useful for the /health endpoint)."""
        if not self._available:
            return {"redis": "unavailable"}
        try:
            info = self._client.info("stats")
            keyspace = self._client.info("keyspace")
            return {
                "redis": "connected",
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate_pct": round(
                    100 * info.get("keyspace_hits", 0)
                    / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1),
                    1,
                ),
                "total_keys": sum(
                    v.get("keys", 0) for v in keyspace.values()
                    if isinstance(v, dict)
                ),
            }
        except Exception as exc:
            return {"redis": f"error: {exc}"}

    # ── Layer 2a: User profile ────────────────────────────────────────────────

    def get_profile(self, user_id: str) -> Optional[dict]:
        """Return user profile from Redis, falling back to diskcache."""
        if self._available:
            data = self._rget(f"user:{user_id}:profile")
            if data is not None:
                return data
        return self._fallback.get_profile(user_id) if self._fallback else None

    def set_profile(self, user_id: str, profile: dict) -> None:
        """Store user profile in Redis AND diskcache for dual redundancy."""
        if self._available:
            self._rsetex(f"user:{user_id}:profile", self._profile_ttl, profile)
        if self._fallback:
            self._fallback.set_profile(user_id, profile)

    # ── Layer 2b: Query history ───────────────────────────────────────────────

    def get_query_history(self, user_id: str) -> list:
        """Return query history (list of {prompt, pandas_operation, result_summary})."""
        if self._available:
            data = self._rget(f"user:{user_id}:query_history")
            if data is not None:
                return data if isinstance(data, list) else []
        return self._fallback.get_query_history(user_id) if self._fallback else []

    def append_query_history(
        self,
        user_id: str,
        prompt: str,
        pandas_operation: str,
        result_summary: str,
    ) -> None:
        """
        Append a (prompt, pandas_operation, result_summary) tuple to the
        query history, capped at MAX_QUERY_HISTORY entries.
        """
        history = self.get_query_history(user_id)
        history.append({
            "prompt": prompt,
            "pandas_operation": pandas_operation,
            "result_summary": result_summary,
        })
        history = history[-self._max_history:]
        if self._available:
            self._rsetex(f"user:{user_id}:query_history", self._history_ttl, history)
        if self._fallback:
            # Keep diskcache in sync (fallback layer)
            self._fallback.append_query_history(
                user_id, prompt, pandas_operation, result_summary
            )

    # ── Layer 2c: Viz state ───────────────────────────────────────────────────

    def get_viz_state(self, user_id: str) -> Optional[dict]:
        """Return viz state (last chart type, axes, filters)."""
        if self._available:
            data = self._rget(f"user:{user_id}:viz_state")
            if data is not None:
                return data
        return self._fallback.get_viz_state(user_id) if self._fallback else None

    def set_viz_state(self, user_id: str, state: dict) -> None:
        """Store viz state in Redis AND diskcache."""
        if self._available:
            self._rsetex(f"user:{user_id}:viz_state", self._viz_ttl, state)
        if self._fallback:
            self._fallback.set_viz_state(user_id, state)

    # ── Circuit-breaker delegation ────────────────────────────────────────────
    # Always delegate to diskcache — circuit-breaker state must survive
    # Redis restarts, so we keep it on durable diskcache only.

    def is_circuit_open(self) -> bool:
        return self._fallback.is_circuit_open() if self._fallback else False

    def record_circuit_failure(self) -> bool:
        return self._fallback.record_circuit_failure() if self._fallback else False

    def record_circuit_success(self) -> None:
        if self._fallback:
            self._fallback.record_circuit_success()
