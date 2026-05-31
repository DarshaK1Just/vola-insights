"""
LangGraph DAG for the DataFrame-first financial AI pipeline.

Flow:
  validate_user
    -> [blocked] compose_response -> END
    -> input_guardrail
        -> [blocked] compose_response -> END
        -> fetch_profile
            -> llm_reason
                -> [LLM_FAILURE] graceful_degradation -> output_guardrail -> compose_response -> END
                -> output_guardrail -> compose_response -> END
"""
import logging

import pandas as pd
from langgraph.graph import StateGraph, END

from src.graph.state import RAGState
from src.graph.nodes import (
    make_validate_user_node,
    make_input_guardrail_node,
    make_fetch_profile_node,
    make_llm_reason_node,
    make_output_guardrail_node,
    make_compose_response_node,
    make_graceful_degradation_node,
)

logger = logging.getLogger(__name__)


def build_graph(df: pd.DataFrame, cache, input_guard, output_guard, audit):
    """
    Build and compile the LangGraph pipeline.

    Parameters
    ----------
    df          : Full transactions DataFrame.
    cache       : UserCacheManager instance.
    input_guard : InputGuard instance.
    output_guard: OutputGuard instance.
    audit       : AuditLogger instance.
    """
    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("validate_user",       make_validate_user_node(df))
    graph.add_node("input_guardrail",     make_input_guardrail_node(input_guard))
    graph.add_node("fetch_profile",       make_fetch_profile_node(cache))
    graph.add_node("llm_reason",          make_llm_reason_node(cache))
    graph.add_node("output_guardrail",    make_output_guardrail_node(output_guard))
    graph.add_node("compose_response",    make_compose_response_node(cache, audit))
    graph.add_node("graceful_degradation", make_graceful_degradation_node(cache, audit))

    # Entry point
    graph.set_entry_point("validate_user")

    # validate_user: error (invalid user_id) -> compose_response, else -> input_guardrail
    graph.add_conditional_edges(
        "validate_user",
        lambda s: "compose_response" if s.get("blocked") or s.get("error") else "input_guardrail",
        {"compose_response": "compose_response", "input_guardrail": "input_guardrail"},
    )

    # input_guardrail: blocked -> compose_response, else -> fetch_profile
    graph.add_conditional_edges(
        "input_guardrail",
        lambda s: "compose_response" if s.get("blocked") else "fetch_profile",
        {"compose_response": "compose_response", "fetch_profile": "fetch_profile"},
    )

    # fetch_profile -> llm_reason
    graph.add_edge("fetch_profile", "llm_reason")

    # llm_reason: LLM failure -> graceful_degradation, else -> output_guardrail
    graph.add_conditional_edges(
        "llm_reason",
        lambda s: (
            "graceful_degradation"
            if s.get("error") and str(s.get("error", "")).startswith("LLM_FAILURE")
            else "output_guardrail"
        ),
        {"graceful_degradation": "graceful_degradation", "output_guardrail": "output_guardrail"},
    )

    # graceful_degradation -> output_guardrail (still validate the fallback response)
    graph.add_edge("graceful_degradation", "output_guardrail")

    # output_guardrail -> compose_response -> END
    graph.add_edge("output_guardrail", "compose_response")
    graph.add_edge("compose_response", END)

    return graph.compile()
