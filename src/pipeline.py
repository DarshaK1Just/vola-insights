"""
TransactionRAGPipeline — public interface.

Startup strategy
────────────────
__init__ returns in < 200 ms:
  • DataFrame copy + UserCacheManager + AuditLogger — all instant
  • A background daemon thread is started immediately after

Background thread (runs while API already serves /health & /users):
  • InputGuard()   — loads guardrails-ai (~20-30 s on first run)
  • OutputGuard()  — loads guardrails-ai
  • RedisSemanticCache() — connects to Redis
  • build_graph()  — compiles LangGraph DAG

run() behaviour:
  • While background thread is still running → waits (first caller blocks)
  • Once ready → full pipeline OR Redis semantic cache hit (<1 ms)

Output contract (assessment spec):
    {
        "user_name":       str,
        "response":        str,
        "data_summary":    dict,
        "visualizations":  list[str],
        "cache_hit":       bool,
        "latency_ms":      float,
        "guardrail_flags": list[str],
        "error":           str | None,
    }
"""
import time
import logging
import threading
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from src.config import Config
from src.cache import UserCacheManager
from src.audit_logger import AuditLogger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

logger = logging.getLogger(__name__)


class TransactionRAGPipeline:
    """
    DataFrame-first financial AI pipeline with two-layer caching.

    __init__ is intentionally fast (~200 ms). Heavy components
    (guardrails, Redis, LangGraph) are initialised in a background thread.
    """

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self._all_user_ids = (
            list(df["user_id"].unique()) if "user_id" in df.columns else []
        )

        # ── Fast components (synchronous) ────────────────────────────────────
        self._disk_cache = UserCacheManager(Config.CACHE_DIR, Config)
        self._audit      = AuditLogger(Config.LOG_DIR)

        # Heavy components — initialised in background
        self._input_guard  = None
        self._output_guard = None
        self._cache        = self._disk_cache   # diskcache until Redis is ready
        self._graph        = None

        # Event that .run() waits on — set when background init finishes
        self._ready = threading.Event()
        self._init_error: str = ""

        threading.Thread(
            target=self._background_init,
            daemon=True,
            name="pipeline-init",
        ).start()

        logger.info(
            "TransactionRAGPipeline created — %d rows, %d users "
            "(heavy components loading in background...)",
            len(self._df), len(self._all_user_ids),
        )

    def _background_init(self) -> None:
        """
        Initialise everything that's slow in a background daemon thread.
        Order matters: Guards first (longest), then Redis, then LangGraph.
        """
        t0 = time.time()
        try:
            # 1. Guardrails (slowest — loads guardrails-ai, may ~20-50 s first run)
            logger.info("[bg-init] Loading guardrails...")
            from src.guardrails.input_guard import InputGuard
            from src.guardrails.output_guard import OutputGuard
            self._input_guard  = InputGuard()
            self._output_guard = OutputGuard()
            logger.info("[bg-init] Guardrails ready (%.1f s)", time.time() - t0)

            # 2. Redis semantic cache (fast ~100-300 ms, falls back if unavailable)
            logger.info("[bg-init] Connecting Redis cache...")
            from src.redis_cache import RedisSemanticCache
            redis_cache = RedisSemanticCache(
                redis_url=Config.REDIS_URL,
                response_ttl=Config.RESPONSE_CACHE_TTL,
                profile_ttl=Config.PROFILE_TTL,
                history_ttl=Config.QUERY_HISTORY_TTL,
                viz_ttl=Config.VIZ_STATE_TTL,
                max_history=Config.MAX_QUERY_HISTORY,
                fallback_cache=self._disk_cache if Config.REDIS_ENABLED else None,
            ) if Config.REDIS_ENABLED else self._disk_cache
            self._cache = redis_cache
            redis_status = (
                "Redis ON" if (
                    Config.REDIS_ENABLED
                    and isinstance(redis_cache, RedisSemanticCache)
                    and redis_cache.available
                ) else "Redis OFF (diskcache)"
            )
            logger.info("[bg-init] Cache ready — %s (%.1f s)", redis_status, time.time() - t0)

            # 3. LangGraph DAG compilation (fast ~100-500 ms)
            logger.info("[bg-init] Compiling LangGraph pipeline...")
            from src.graph.builder import build_graph
            self._graph = build_graph(
                df=self._df,
                cache=self._cache,
                input_guard=self._input_guard,
                output_guard=self._output_guard,
                audit=self._audit,
            )
            logger.info("[bg-init] LangGraph ready (%.1f s)", time.time() - t0)

            # Pre-warm: compute and cache profiles for all users in background
            self._prewarm_profiles()

        except Exception as exc:
            self._init_error = str(exc)
            logger.error("[bg-init] FAILED: %s", exc, exc_info=True)
        finally:
            elapsed = time.time() - t0
            if self._init_error:
                logger.error("[bg-init] Pipeline init failed after %.1f s", elapsed)
            else:
                logger.info(
                    "TransactionRAGPipeline FULLY READY — %d rows, %d users "
                    "| total init: %.1f s",
                    len(self._df), len(self._all_user_ids), elapsed,
                )
            self._ready.set()   # unblock any waiting .run() calls

    def _prewarm_profiles(self) -> None:
        """Pre-compute and cache user profiles so first queries feel instant."""
        from src.data_loader import compute_user_profile, get_user_data
        warmed = 0
        for uid in self._all_user_ids:
            if self._cache.get_profile(uid) is None:
                try:
                    user_df, uname = get_user_data(self._df, uid)
                    profile = compute_user_profile(user_df, uid, uname)
                    self._cache.set_profile(uid, profile)
                    warmed += 1
                except Exception as exc:
                    logger.warning("[bg-init] Profile prewarm failed for %s: %s", uid, exc)
        if warmed:
            logger.info("[bg-init] Pre-warmed %d user profile(s)", warmed)

    # ── Public: run ───────────────────────────────────────────────────────────

    def run(self, user_id: str, prompt: str) -> dict:
        """
        Execute the pipeline.

        If background init hasn't finished yet, this call blocks until it does
        (max 120 s). All subsequent calls are non-blocking once ready.
        """
        start = time.time()

        # Block until background init completes (only relevant for the very
        # first request that arrives while guardrails are still loading)
        if not self._ready.is_set():
            logger.info("Waiting for pipeline background init to complete...")
            self._ready.wait(timeout=120)

        # Surface init errors
        if self._init_error:
            return {
                "user_name": user_id,
                "response":  f"Pipeline initialisation failed: {self._init_error}",
                "data_summary": {},
                "visualizations": [],
                "cache_hit": False,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "guardrail_flags": ["INIT_ERROR"],
                "error": self._init_error,
            }

        # ── Circuit breaker ───────────────────────────────────────────────────
        if self._cache.is_circuit_open():
            logger.warning("Circuit breaker open — returning cached fallback")
            profile = self._cache.get_profile(user_id) or {}
            return {
                "user_name": profile.get("user_name", user_id),
                "response": (
                    "The AI service is temporarily unavailable "
                    "(circuit breaker open). Please try again in a few minutes."
                ),
                "data_summary": profile,
                "visualizations": [],
                "cache_hit": False,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "guardrail_flags": ["CIRCUIT_OPEN"],
                "error": "circuit_open",
            }

        # ── Fast path: Redis semantic cache ───────────────────────────────────
        from src.redis_cache import RedisSemanticCache
        if isinstance(self._cache, RedisSemanticCache) and self._cache.available:
            cached = self._cache.get_response(user_id, prompt)
            if cached is not None:
                # Cache hit - but regenerate charts for fresh visualizations
                logger.info(
                    "Semantic cache hit for user=%s - regenerating charts",
                    user_id,
                )
                
                # Get cached tool calls to regenerate charts
                cached_tool_calls = cached.get("cached_tool_calls", [])
                chart_paths = []
                
                if cached_tool_calls:
                    from src.visualizations import execute_viz_tool, VIZ_TOOL_NAMES
                    from src.data_loader import get_user_data
                    
                    # Get user data for chart generation
                    user_df, user_name = get_user_data(self._df, user_id)
                    
                    # Regenerate only visualization tools
                    for tc in cached_tool_calls:
                        if tc.get("name") in VIZ_TOOL_NAMES:
                            try:
                                path = execute_viz_tool(
                                    tc["name"], 
                                    tc.get("args", {}), 
                                    user_df, 
                                    user_id, 
                                    user_name
                                )
                                if path:
                                    chart_paths.append(path)
                                    logger.info("Regenerated chart: %s", path)
                            except Exception as e:
                                logger.warning("Failed to regenerate chart %s: %s", tc["name"], e)
                
                # Update cached response with fresh charts
                cached["visualizations"] = chart_paths
                cached["cache_hit"] = True
                cached["latency_ms"] = round((time.time() - start) * 1000, 1)
                
                logger.info(
                    "Cache hit for user=%s with %d regenerated charts (%.1f ms)",
                    user_id, len(chart_paths), cached["latency_ms"],
                )
                return cached

        # ── Slow path: full LangGraph pipeline ───────────────────────────────
        initial_state = {
            "user_id": user_id,
            "prompt": prompt,
            "user_name": "",
            "user_df": None,
            "all_user_ids": [],
            "profile": {},
            "cache_hit": False,
            "tool_calls": [],
            "chart_paths": [],
            "llm_messages": [],
            "llm_response": "",
            "final_response": "",
            "data_summary": {},
            "guardrail_flags": [],
            "error": None,
            "blocked": False,
            "blocked_response": None,
            "model_used": Config.MODEL_PRIMARY,
            "start_time": start,
            "latency_ms": 0.0,
        }

        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.error("Graph error for user=%s: %s", user_id, exc, exc_info=True)
            self._cache.record_circuit_failure()
            safe_profile = self._cache.get_profile(user_id) or {}
            return {
                "user_name": safe_profile.get("user_name", user_id),
                "response": "An unexpected pipeline error occurred. Please try again.",
                "data_summary": safe_profile,
                "visualizations": [],
                "cache_hit": False,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "guardrail_flags": ["PIPELINE_ERROR"],
                "error": str(exc),
            }

        if not final_state.get("error"):
            self._cache.record_circuit_success()
        elif str(final_state.get("error", "")).startswith("LLM_FAILURE"):
            self._cache.record_circuit_failure()

        result = {
            "user_name": final_state.get("user_name", user_id),
            "response": final_state.get("final_response", ""),
            "data_summary": final_state.get("data_summary", {}),
            "visualizations": final_state.get("chart_paths", []),
            "cache_hit": final_state.get("cache_hit", False),
            "latency_ms": final_state.get("latency_ms", round((time.time() - start) * 1000, 1)),
            "guardrail_flags": final_state.get("guardrail_flags", []),
            "error": final_state.get("error"),
            "tool_calls": final_state.get("tool_calls", []),  # Store for chart regeneration
        }

        if isinstance(self._cache, RedisSemanticCache) and self._cache.available:
            self._cache.set_response(user_id, prompt, result)

        return result

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def users(self) -> list:
        """Available immediately — reads directly from the DataFrame."""
        if "user_id" not in self._df.columns:
            return []
        result = []
        for uid, grp in self._df.groupby("user_id"):
            uname = str(grp["user_name"].iloc[0]) if "user_name" in grp.columns else uid
            result.append({"user_id": uid, "user_name": uname})
        return sorted(result, key=lambda x: x["user_name"])

    @property
    def is_ready(self) -> bool:
        """True once background init has finished."""
        return self._ready.is_set()

    @property
    def cache_info(self) -> dict:
        """Returns cache stats; available immediately."""
        from src.redis_cache import RedisSemanticCache
        if isinstance(self._cache, RedisSemanticCache):
            return self._cache.cache_stats()
        if not self._ready.is_set():
            return {"redis": "initializing"}
        return {"redis": "disabled", "backend": "diskcache"}
