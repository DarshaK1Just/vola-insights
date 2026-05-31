"""
User-specific KV cache backed by diskcache.

Cache entries per assessment spec:
  user:{id}:profile       — Name, date range, top categories, avg monthly spend
  user:{id}:query_history — List of {prompt, pandas_operation, result_summary} dicts
  user:{id}:viz_state     — Last chart type, axes, filters (visualization continuity)
"""
import time
import logging

import diskcache

logger = logging.getLogger(__name__)


class UserCacheManager:
    def __init__(self, cache_dir: str, config):
        self._cache = diskcache.Cache(cache_dir)
        self._config = config

    # ── Profile ───────────────────────────────────────────────────────────────

    def get_profile(self, user_id: str) -> dict | None:
        return self._cache.get(f"user:{user_id}:profile", default=None)

    def set_profile(self, user_id: str, profile: dict) -> None:
        self._cache.set(f"user:{user_id}:profile", profile, expire=self._config.PROFILE_TTL)

    # ── Query history — stores (prompt, pandas_operation, result_summary) ─────

    def get_query_history(self, user_id: str) -> list:
        """Return list of {prompt, pandas_operation, result_summary} dicts."""
        return self._cache.get(f"user:{user_id}:query_history", default=[])

    def append_query_history(
        self,
        user_id: str,
        prompt: str,
        pandas_operation: str,
        result_summary: str,
    ) -> None:
        """
        Append a new (prompt, pandas_operation, result_summary) tuple to the
        user's query history and cap at MAX_QUERY_HISTORY entries.
        """
        key = f"user:{user_id}:query_history"
        history: list = self._cache.get(key, default=[])
        history.append({
            "prompt": prompt,
            "pandas_operation": pandas_operation,
            "result_summary": result_summary,
        })
        history = history[-self._config.MAX_QUERY_HISTORY:]
        self._cache.set(key, history, expire=self._config.QUERY_HISTORY_TTL)

    # ── Viz state ─────────────────────────────────────────────────────────────

    def get_viz_state(self, user_id: str) -> dict | None:
        return self._cache.get(f"user:{user_id}:viz_state", default=None)

    def set_viz_state(self, user_id: str, state: dict) -> None:
        self._cache.set(f"user:{user_id}:viz_state", state, expire=self._config.VIZ_STATE_TTL)

    # ── Circuit breaker ───────────────────────────────────────────────────────

    _CIRCUIT_KEY = "circuit_breaker:state"
    _CIRCUIT_DEFAULT: dict = {"open": False, "failures": 0, "opened_at": 0.0}

    def get_circuit_state(self) -> dict:
        return self._cache.get(self._CIRCUIT_KEY, default=dict(self._CIRCUIT_DEFAULT))

    def record_circuit_failure(self) -> bool:
        state = self.get_circuit_state()
        state["failures"] += 1
        just_opened = False
        if (
            state["failures"] >= self._config.CIRCUIT_BREAKER_THRESHOLD
            and not state["open"]
        ):
            state["open"] = True
            state["opened_at"] = time.time()
            just_opened = True
            logger.warning("Circuit breaker opened after %d failures.", state["failures"])
        self._cache.set(self._CIRCUIT_KEY, state, expire=self._config.CIRCUIT_TTL)
        return just_opened

    def record_circuit_success(self) -> None:
        self._cache.set(self._CIRCUIT_KEY, dict(self._CIRCUIT_DEFAULT), expire=self._config.CIRCUIT_TTL)

    def is_circuit_open(self) -> bool:
        state = self.get_circuit_state()
        if not state["open"]:
            return False
        elapsed = time.time() - state["opened_at"]
        if elapsed >= self._config.CIRCUIT_RESET_SECONDS:
            logger.info("Circuit breaker auto-reset after %.1f seconds.", elapsed)
            self.record_circuit_success()
            return False
        return True
