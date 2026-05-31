"""LangGraph state TypedDict for the DataFrame-first financial AI pipeline."""
from typing import Any, Optional
from typing_extensions import TypedDict


class RAGState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    user_id: str
    prompt: str

    # ── User data ─────────────────────────────────────────────────────────────
    user_name: str
    user_df: Any           # pd.DataFrame (user-filtered)
    all_user_ids: list     # all user IDs in the system (for cross-user leak check)

    # ── Cache & profile ───────────────────────────────────────────────────────
    profile: dict
    cache_hit: bool

    # ── Tool execution results ────────────────────────────────────────────────
    tool_calls: list       # [{name, args, result, id}] — analysis + viz
    chart_paths: list      # paths to saved PNG files

    # ── LLM interaction ───────────────────────────────────────────────────────
    llm_messages: list
    llm_response: str      # final synthesised text

    # ── Output ───────────────────────────────────────────────────────────────
    final_response: str
    data_summary: dict
    guardrail_flags: list

    # ── Control flow ─────────────────────────────────────────────────────────
    error: Optional[str]
    blocked: bool
    blocked_response: Optional[str]

    # ── LLM model actually used (PRIMARY or a fallback) ──────────────────────
    model_used: str

    # ── Timing ───────────────────────────────────────────────────────────────
    start_time: float
    latency_ms: float
