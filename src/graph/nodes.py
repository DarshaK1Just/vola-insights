# -*- coding: utf-8 -*-
"""
LangGraph node factory functions — DataFrame-first financial AI pipeline.

Architecture — DataFrame-first, pandas tool calling, OpenRouter LLM:
  validate_user       — validates user_id, loads user_df from master DataFrame
  input_guardrail     — injection / cross-user / scope / length checks
  fetch_profile       — load or compute user profile from diskcache
  llm_reason          — TWO-phase LLM call:
                          Phase 1: LLM picks analysis tools + viz tools to call
                          Phase 2: LLM synthesises final text from tool results
  output_guardrail    — toxicity / cross-user / hallucination checks
  compose_response    — build final output, update cache with pandas_operation tuple
  graceful_degradation— fallback when LLM is unreachable
"""
import json
import logging
import time
import re
from datetime import datetime

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from src.graph.state import RAGState
from src.cache import UserCacheManager
from src.data_loader import compute_user_profile, get_user_data
from src.guardrails.input_guard import InputGuard
from src.guardrails.output_guard import OutputGuard
from src.analysis_tools import (
    get_analysis_tool_schemas,
    execute_analysis_tool,
    format_tool_result_for_llm,
    summarise_tool_result,
    ANALYSIS_TOOL_NAMES,
)
from src.visualizations import (
    get_viz_tool_schemas,
    execute_viz_tool,
    VIZ_TOOL_NAMES,
)
from src.audit_logger import AuditLogger
from src.config import Config

logger = logging.getLogger(__name__)


# ── Unicode normalizer (keep for response cleanup) ────────────────────────────

_UNICODE_SPACES = [chr(c) for c in [0x00A0, 0x2000, 0x2001, 0x2002, 0x2003,
    0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x202F, 0x205F, 0x3000]]
_SPECIAL_DASHES = [chr(c) for c in [0x2010, 0x2011, 0x2013, 0x2014, 0x2015, 0x2212]]
_ZERO_WIDTH = [chr(c) for c in [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF]]


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    for ch in _UNICODE_SPACES:
        text = text.replace(ch, " ")
    for ch in _SPECIAL_DASHES:
        text = text.replace(ch, "-")
    for ch in _ZERO_WIDTH:
        text = text.replace(ch, "")
    text = text.replace(chr(0x2018), "'").replace(chr(0x2019), "'")
    text = text.replace(chr(0x201C), '"').replace(chr(0x201D), '"')
    return text


# ── LLM factory & resilient caller ───────────────────────────────────────────

def _make_llm(model: str = None) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    return ChatOpenAI(
        model=model or Config.MODEL_PRIMARY,
        openai_api_base=Config.OPENROUTER_BASE_URL,
        openai_api_key=Config.OPENROUTER_API_KEY,
        temperature=0.1,
        max_tokens=Config.MAX_OUTPUT_TOKENS,
        timeout=Config.TIMEOUT_SECONDS,
        max_retries=1,           # HTTP-level retries per model attempt
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Vola Insights",
        },
    )


def _call_llm_with_fallback(
    messages: list,
    tools: list = None,
    *,
    label: str = "LLM",
) -> tuple:
    """
    Call the LLM with explicit exponential backoff and model fallback.

    Tries models in order: PRIMARY → FALLBACK_1 → FALLBACK_2.
    Waits BACKOFF_BASE^attempt seconds between attempts.

    Returns (response, model_name_used).
    Raises the last exception if all models fail.
    """
    fallback_models = [
        Config.MODEL_PRIMARY,
        Config.MODEL_FALLBACK_1,
        Config.MODEL_FALLBACK_2,
    ]
    last_exc: Exception = RuntimeError("No models configured")

    for attempt, model in enumerate(fallback_models):
        if attempt > 0:
            backoff = Config.BACKOFF_BASE ** attempt   # 2s, 4s
            logger.warning(
                "%s: model %s failed, trying fallback %s in %.1fs (attempt %d/%d)",
                label, fallback_models[attempt - 1], model, backoff, attempt + 1, len(fallback_models),
            )
            time.sleep(backoff)

        try:
            llm = _make_llm(model)
            if tools:
                llm = llm.bind_tools(tools)
            response = llm.invoke(messages)
            if attempt > 0:
                logger.info("%s: fallback model %s succeeded", label, model)
            return response, model
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc)
            logger.warning("%s attempt %d/%d failed: %s", label, attempt + 1, len(fallback_models), exc_str[:120])
            # Auth errors are permanent — no point trying other models
            if "401" in exc_str or "unauthorized" in exc_str.lower():
                logger.error("%s: auth error, stopping fallback chain", label)
                break

    raise last_exc


# ── Context helpers ───────────────────────────────────────────────────────────

def _profile_to_text(profile: dict) -> str:
    if not profile:
        return "No profile data available."
    monthly = profile.get("monthly_summary", {})
    monthly_lines = ""
    if monthly:
        sorted_months = sorted(monthly.items())
        monthly_lines = "\nMonthly breakdown:\n" + "\n".join(
            f"  {m}: income ${v.get('income', 0):,.0f} | "
            f"expenses ${v.get('expenses', 0):,.0f} | "
            f"net ${v.get('net', 0):+,.0f}"
            for m, v in sorted_months
        )
    top_cats = profile.get("top_expense_categories", [])
    cats_str = (
        ", ".join(f"{c['category']} ${c['total']:,.0f}" for c in top_cats[:5])
        if top_cats else "N/A"
    )
    return (
        f"Data range: {profile.get('date_range', {}).get('start', '?')} "
        f"to {profile.get('date_range', {}).get('end', '?')}\n"
        f"Total transactions: {profile.get('total_transactions', 0)}\n"
        f"Total income: ${profile.get('total_income', 0):,.2f}\n"
        f"Total expenses: ${profile.get('total_expenses', 0):,.2f}\n"
        f"Net savings: ${profile.get('net_savings', 0):,.2f}\n"
        f"Avg monthly spend: ${profile.get('avg_monthly_spend', 0):,.2f}\n"
        f"Top expense categories: {cats_str}"
        f"{monthly_lines}"
    )


def _history_to_few_shots(history: list) -> str:
    if not history:
        return ""
    examples = []
    for h in history[-Config.FEW_SHOT_HISTORY_N:]:
        q = h.get("prompt", "")
        op = h.get("pandas_operation", "")
        r = h.get("result_summary", "")[:200]
        if q:
            op_hint = f" [operation: {op}]" if op else ""
            examples.append(f"Q{op_hint}: {q}\nA: {r}")
    if not examples:
        return ""
    return (
        "\n\nPrevious interaction history (few-shot examples from this user):\n"
        + "\n---\n".join(examples)
    )


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _truncate_to_budget(text: str, budget_tokens: int) -> str:
    max_words = int(budget_tokens / 1.3)
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "\n[... context truncated to fit token budget]"
    return text


# ── NODE FACTORIES ────────────────────────────────────────────────────────────

def make_validate_user_node(df: pd.DataFrame):
    all_user_ids = df["user_id"].unique().tolist() if "user_id" in df.columns else []

    def validate_user(state: RAGState) -> dict:
        user_id = state["user_id"]
        if user_id not in df["user_id"].values:
            return {
                "error": f"User '{user_id}' not found.",
                "blocked": True,
                "blocked_response": f"User ID '{user_id}' does not exist in the system.",
                "all_user_ids": all_user_ids,
            }
        user_df, user_name = get_user_data(df, user_id)
        return {
            "user_df": user_df,
            "user_name": user_name,
            "all_user_ids": all_user_ids,
            "error": None,
            "blocked": False,
        }

    return validate_user


def make_input_guardrail_node(input_guard: InputGuard):
    def input_guardrail(state: RAGState) -> dict:
        prompt = state["prompt"]
        user_id = state.get("user_id", "")
        result = input_guard.check(prompt, user_id)

        updates: dict = {
            "blocked": not result.passed,
            "blocked_response": result.blocked_response or "",
            "guardrail_flags": result.flags,
        }
        # Use truncated prompt if length was enforced
        if result.passed and result.effective_prompt and result.effective_prompt != prompt:
            updates["prompt"] = result.effective_prompt

        return updates

    return input_guardrail


def make_fetch_profile_node(cache: UserCacheManager):
    def fetch_profile(state: RAGState) -> dict:
        user_id = state["user_id"]
        profile = cache.get_profile(user_id)
        cache_hit = profile is not None
        if not cache_hit:
            user_df = state["user_df"]
            user_name = state["user_name"]
            try:
                profile = compute_user_profile(user_df, user_id, user_name)
                cache.set_profile(user_id, profile)
            except Exception as exc:
                logger.error("Profile compute failed for %s: %s", user_id, exc)
                profile = {"user_id": user_id, "user_name": user_name}
        return {"profile": profile, "cache_hit": cache_hit}

    return fetch_profile


def make_llm_reason_node(cache: UserCacheManager):
    """
    Two-phase LLM node — DataFrame-first, no RAG.

    Phase 1:
      LLM receives: user profile + data schema + few-shot history + user prompt + ALL tools
      LLM calls:    any subset of analysis tools (pandas ops) + viz tools (including generate_dynamic_chart)

    Execute all tool calls (analysis + viz)

    Phase 2:
      LLM receives: tool results → synthesises a natural language response
    """
    all_tool_schemas = get_analysis_tool_schemas() + get_viz_tool_schemas()

    SCHEMA_DESCRIPTION = (
        "Transaction DataFrame columns:\n"
        "  user_id                     (str)     Unique user identifier\n"
        "  user_name                   (str)     Display name\n"
        "  transaction_date            (datetime) When the transaction occurred\n"
        "  transaction_amount          (float)   NEGATIVE = income, POSITIVE = expense\n"
        "  merchant_name               (str)     Merchant or employer name\n"
        "  transaction_category_detail (str)     Hierarchical: e.g. Food > Restaurants > Fast Food"
    )

    # Explicit chart-selection rules injected into every system prompt
    CHART_SELECTION_GUIDE = """\
AUTONOMOUS CHART SELECTION — follow these rules exactly:

RULE 1 — "full financial report" / "give me a full report" / "full overview":
  MUST call ALL FOUR charts:
    plot_monthly_spending_trend(user_id, months=12)
    plot_category_breakdown(user_id, period="last_3_months", top_n=7)
    plot_income_vs_expense(user_id, months=6, show_net_line=True)
    generate_dynamic_chart(user_id, chart_type="bar", title="Monthly Savings Rate",
                           data_source="savings_rate", y_label="Savings Rate (%)")
  ALSO call get_monthly_trend(months=12) and get_spending_by_category(months=3) for the text summary.

RULE 2 — "how am I doing financially?" / "financial health":
  call plot_income_vs_expense(user_id, months=6) + plot_category_breakdown(user_id)
  call get_period_stats(period="last_3_months") for the text.

RULE 3 — "spending trend" / "how has spending changed" / "monthly pattern":
  call plot_monthly_spending_trend(user_id, months=12)
  call get_monthly_trend(months=12) for the text.

RULE 4 — "where is my money going" / "spending categories" / "what do I spend most on":
  call plot_category_breakdown(user_id, period="last_3_months")
  call get_spending_by_category(months=3) for the text.

RULE 5 — "am I saving" / "income vs expenses" / "savings":
  call plot_income_vs_expense(user_id, months=6, show_net_line=True)
  call get_income_analysis(months=6) for the text.

RULE 6 — "food spending" / "show me [CATEGORY] spending" / "[category] breakdown":
  The spec requires: top_subcategories with parent_category=[CATEGORY]
  MUST call:
    plot_monthly_spending_trend(user_id, months=6, category_filter="[CATEGORY]")
    generate_dynamic_chart(user_id, chart_type="bar",
                           title="[CATEGORY] Subcategory Breakdown",
                           data_source="subcategory_totals",
                           parent_category="[CATEGORY]")
  ALSO call get_spending_by_category(months=3) for the text numbers.
  Examples:
    "Show me my food spending"    → parent_category="Food"
    "Show me transport spending"  → parent_category="Transport"
    "My entertainment costs"      → parent_category="Entertainment"

RULE 7 — "top merchants" / "where am I shopping":
  call get_merchant_analysis(months=3)
  call generate_dynamic_chart(user_id, chart_type="bar", title="Top Merchants by Spend",
                               data_source="merchant_totals", months=3)

RULE 8 — simple factual questions ("what did I spend last month"):
  call get_period_stats(period="last_month") for the exact number.
  call plot_category_breakdown(user_id, period="last_month") to visualise.

generate_dynamic_chart data_source quick-reference:
  monthly_expenses    → bar/line: monthly spend over time
  monthly_income      → bar/line: monthly income over time
  category_totals     → bar/donut: top-level spending categories
  subcategory_totals  → bar: subcategories under a parent (REQUIRES parent_category="Food" etc.)
  merchant_totals     → bar: top merchants
  savings_rate        → bar/line: monthly savings % (use y_label="Savings Rate (%)")
  monthly_net_savings → bar/line: monthly net savings in dollars
  custom              → LLM provides data=[{label, value}] directly
"""

    def llm_reason(state: RAGState) -> dict:
        user_id   = state["user_id"]
        user_name = state["user_name"]
        prompt    = state["prompt"]
        profile   = state.get("profile", {})
        user_df   = state["user_df"]

        # ── Build system context ─────────────────────────────────────────────
        profile_text = _profile_to_text(profile)
        history      = cache.get_query_history(user_id)
        few_shots    = _history_to_few_shots(history)
        viz_state    = cache.get_viz_state(user_id)
        viz_hint     = ""
        if viz_state:
            last = viz_state.get("last_chart_types", [])
            if last:
                viz_hint = (
                    f"\nVisualization continuity: last session used "
                    f"{', '.join(last)}. Complement rather than duplicate."
                )

        # Token budget enforcement
        context_text = profile_text + SCHEMA_DESCRIPTION + few_shots
        token_est = _estimate_tokens(context_text)
        if token_est > Config.MAX_INPUT_TOKENS - 1000:
            few_shots = _truncate_to_budget(few_shots, max(500, Config.MAX_INPUT_TOKENS - 2000))
            logger.warning("Context truncated: ~%d tokens (budget %d)", token_est, Config.MAX_INPUT_TOKENS)

        today_str = datetime.now().strftime("%B %d, %Y")
        system_content = (
            f"You are a precise personal financial analyst assistant for {user_name}.\n"
            f"Today is {today_str}.\n\n"
            "CRITICAL RULES:\n"
            "1. ONLY discuss THIS user's own data. Never reference other users.\n"
            "2. Call analysis tools to get accurate numerical data BEFORE answering.\n"
            "3. Call visualization tools to generate charts when they add insight.\n"
            "4. If data is insufficient for a claim, clearly say so.\n"
            "5. NEGATIVE amounts = income received. POSITIVE amounts = money spent.\n\n"
            f"{SCHEMA_DESCRIPTION}\n\n"
            f"USER FINANCIAL PROFILE:\n{profile_text}"
            f"{few_shots}"
            f"{viz_hint}\n\n"
            f"{CHART_SELECTION_GUIDE}"
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=prompt),
        ]

        executed_tool_calls: list = []
        chart_paths: list = []
        phase1_ai_message = None
        model_used = Config.MODEL_PRIMARY
        error_code = None

        # ── Phase 1: LLM selects tools (with fallback + malformed-JSON retry) ─
        # The doc requires:
        #   - exponential backoff with model fallback (PRIMARY → FALLBACK_1 → FALLBACK_2)
        #   - if the LLM returns unparseable tool calls → retry once, then fall back gracefully
        phase1_response = None
        for parse_attempt in range(2):   # retry ONCE on malformed tool-call JSON
            try:
                response1, model_used = _call_llm_with_fallback(
                    messages, all_tool_schemas, label="Phase1"
                )
                # Validate that every tool-call arg is parseable JSON
                raw = getattr(response1, "tool_calls", None) or []
                for tc in raw:
                    tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    if isinstance(tc_args, str):
                        json.loads(tc_args)   # raises json.JSONDecodeError if malformed
                phase1_response = response1
                break   # success
            except json.JSONDecodeError as parse_exc:
                if parse_attempt == 0:
                    logger.warning(
                        "Phase1: malformed tool-call JSON on first attempt, retrying: %s", parse_exc
                    )
                    # Retry immediately — no backoff needed for parse errors
                else:
                    logger.warning(
                        "Phase1: malformed tool-call JSON after retry, using response anyway"
                    )
                    phase1_response = response1   # use it; args handled gracefully below
            except Exception as exc:
                exc_str = str(exc)
                logger.error("Phase1: all models failed: %s", exc_str)
                if "429" in exc_str or "rate limit" in exc_str.lower():
                    error_code = "LLM_FAILURE:RATE_LIMIT"
                elif "401" in exc_str or "unauthorized" in exc_str.lower():
                    error_code = "LLM_FAILURE:AUTH"
                elif "timeout" in exc_str.lower():
                    error_code = "LLM_FAILURE:TIMEOUT"
                else:
                    error_code = "LLM_FAILURE:UNKNOWN"
                return {"error": error_code, "tool_calls": [], "chart_paths": [], "llm_response": ""}

        phase1_ai_message = phase1_response

        raw_tcs = getattr(response1, "tool_calls", None) or []

        # ── Execute all tool calls ────────────────────────────────────────────
        viz_charts_called: list[str] = []

        for tc in raw_tcs:
            tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
            tc_id   = tc.get("id", tc_name) if isinstance(tc, dict) else getattr(tc, "id", tc_name)

            # Ensure args is a dict (handle malformed JSON string from LLM)
            if isinstance(tc_args, str):
                try:
                    tc_args = json.loads(tc_args)
                except Exception:
                    tc_args = {}

            if tc_name in ANALYSIS_TOOL_NAMES:
                result = execute_analysis_tool(tc_name, tc_args, user_df)
                executed_tool_calls.append(
                    {"name": tc_name, "args": tc_args, "result": result, "id": tc_id}
                )

            elif tc_name in VIZ_TOOL_NAMES:
                # generate_dynamic_chart and fixed tools all routed through execute_viz_tool
                path   = execute_viz_tool(tc_name, tc_args, user_df, user_id, user_name)
                result = {"chart_saved": bool(path), "path": path, "tool": tc_name}
                executed_tool_calls.append(
                    {"name": tc_name, "args": tc_args, "result": result, "id": tc_id}
                )
                if path:
                    chart_paths.append(path)
                    viz_charts_called.append(tc_name)

        # Persist viz_state once after all tool calls
        if viz_charts_called:
            cache.set_viz_state(user_id, {
                "last_chart_types": viz_charts_called,
                "last_prompt":      prompt[:200],
                "chart_paths":      chart_paths,
            })

        # ── Phase 2: LLM synthesises response from tool results ───────────────
        llm_response = ""
        try:
            if executed_tool_calls:
                # Build messages with tool results
                synthesis_messages = list(messages)
                if phase1_ai_message is not None:
                    synthesis_messages.append(phase1_ai_message)
                for tc in executed_tool_calls:
                    synthesis_messages.append(
                        ToolMessage(
                            content=format_tool_result_for_llm(tc["name"], tc["result"]),
                            tool_call_id=tc["id"],
                        )
                    )
                # Second LLM call — synthesise (also uses fallback chain)
                response2, _ = _call_llm_with_fallback(synthesis_messages, label="Phase2")
                llm_response = response2.content if hasattr(response2, "content") else str(response2)
            elif phase1_ai_message is not None and phase1_ai_message.content:
                # No tool calls — LLM answered directly from profile
                llm_response = phase1_ai_message.content
            else:
                llm_response = (
                    "I wasn't able to retrieve specific data to answer your question. "
                    "Please try rephrasing or ask about a different time period."
                )
        except Exception as exc:
            logger.error("LLM Phase 2 failed: %s", exc)
            llm_response = (
                phase1_ai_message.content
                if phase1_ai_message and phase1_ai_message.content
                else f"I encountered an error generating your response. Please try again."
            )

        llm_response = _normalize_text(llm_response)

        # Handle empty response edge case
        if not llm_response.strip():
            profile_fallback = profile
            income = profile_fallback.get("total_income", 0)
            expenses = profile_fallback.get("total_expenses", 0)
            net = profile_fallback.get("net_savings", 0)
            llm_response = (
                f"Based on your financial data:\n"
                f"- Total Income: ${income:,.2f}\n"
                f"- Total Expenses: ${expenses:,.2f}\n"
                f"- Net Savings: ${net:,.2f}\n\n"
                "Please ask a more specific question for a detailed analysis."
            )

        return {
            "tool_calls": executed_tool_calls,
            "chart_paths": chart_paths,
            "llm_response": llm_response,
            "model_used": model_used,
            "error": None,
        }

    return llm_reason


def make_output_guardrail_node(output_guard: OutputGuard):
    def output_guardrail(state: RAGState) -> dict:
        response = state.get("llm_response", "")
        user_id = state.get("user_id", "")
        all_user_ids = state.get("all_user_ids", [])
        profile = state.get("profile", {})
        tool_calls = state.get("tool_calls", [])
        existing_flags = state.get("guardrail_flags", [])

        # Build tool_results dict for precise hallucination checking
        tool_results = {tc["name"]: tc["result"] for tc in tool_calls if "result" in tc}

        result = output_guard.check(
            response=response,
            data_summary=profile,
            user_id=user_id,
            all_user_ids=all_user_ids,
            tool_results=tool_results,
        )
        all_flags = existing_flags + result.flags

        if not result.passed:
            return {
                "blocked": True,
                "blocked_response": result.blocked_response,
                "guardrail_flags": all_flags,
            }
        return {"blocked": False, "guardrail_flags": all_flags}

    return output_guardrail


def make_compose_response_node(cache: UserCacheManager, audit: AuditLogger):
    def compose_response(state: RAGState) -> dict:
        start_time = state.get("start_time", time.time())
        latency_ms = round((time.time() - start_time) * 1000, 1)

        user_id = state["user_id"]
        user_name = state.get("user_name", user_id)
        profile = state.get("profile", {})
        blocked = state.get("blocked", False)
        tool_calls = state.get("tool_calls", [])
        chart_paths = state.get("chart_paths", [])
        guardrail_flags = state.get("guardrail_flags", [])
        prompt = state.get("prompt", "")
        cache_hit = state.get("cache_hit", False)

        # Final response text
        if blocked:
            final_response = state.get("blocked_response") or "Request blocked by safety guardrails."
        else:
            final_response = state.get("llm_response") or "Unable to generate a response."

        final_response = _normalize_text(final_response)

        # Build data_summary
        tool_results = {tc["name"]: tc["result"] for tc in tool_calls if "result" in tc}
        data_summary = {
            "user_name": user_name,
            "total_transactions": profile.get("total_transactions", 0),
            "total_income": profile.get("total_income", 0),
            "total_expenses": profile.get("total_expenses", 0),
            "net_savings": profile.get("net_savings", 0),
            "avg_monthly_spend": profile.get("avg_monthly_spend", 0),
            "date_range": profile.get("date_range", {}),
            "top_expense_categories": profile.get("top_expense_categories", []),
            "tool_results": tool_results,
        }

        # Update query_history with (prompt, pandas_operation, result_summary)
        if not blocked and tool_calls:
            # Primary analysis tool call
            analysis_tcs = [tc for tc in tool_calls if tc["name"] in ANALYSIS_TOOL_NAMES]
            viz_tcs = [tc for tc in tool_calls if tc["name"] in VIZ_TOOL_NAMES]

            if analysis_tcs:
                primary = analysis_tcs[0]
                args_str = ", ".join(f"{k}={v}" for k, v in primary.get("args", {}).items())
                pandas_op = f"{primary['name']}({args_str})"
                result_summary = summarise_tool_result(primary["name"], primary.get("result", {}))
            elif viz_tcs:
                primary = viz_tcs[0]
                args_str = ", ".join(f"{k}={v}" for k, v in primary.get("args", {}).items())
                pandas_op = f"{primary['name']}({args_str})"
                result_summary = f"Generated chart: {primary['name']}"
            else:
                pandas_op = "profile_query()"
                result_summary = final_response[:200]

            cache.append_query_history(
                user_id=user_id,
                prompt=prompt,
                pandas_operation=pandas_op,
                result_summary=result_summary,
            )

        # Audit log — records actual model used (may be a fallback, not primary)
        actual_model = state.get("model_used") or Config.MODEL_PRIMARY
        audit.log_request(
            user_id=user_id,
            prompt=prompt,
            response_length=len(final_response),
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            guardrail_flags=guardrail_flags,
            model_used=actual_model,
        )

        return {
            "final_response": final_response,
            "data_summary": data_summary,
            "latency_ms": latency_ms,
            "chart_paths": chart_paths,
            "guardrail_flags": guardrail_flags,
        }

    return compose_response


def make_graceful_degradation_node(cache: UserCacheManager, audit: AuditLogger):
    """Fallback when LLM is unreachable — uses cached profile data."""

    def graceful_degradation(state: RAGState) -> dict:
        start_time = state.get("start_time", time.time())
        latency_ms = round((time.time() - start_time) * 1000, 1)

        user_id = state["user_id"]
        user_name = state.get("user_name", user_id)
        profile = state.get("profile") or cache.get_profile(user_id) or {}
        error_code = state.get("error", "LLM_FAILURE:UNKNOWN")
        guardrail_flags = list(state.get("guardrail_flags", []))
        prompt = state.get("prompt", "")

        # Human-friendly reason
        if "RATE_LIMIT" in error_code:
            reason = (
                "The AI model has reached its free-tier daily request limit and will "
                "reset at midnight UTC. Here is your cached financial summary:"
            )
        elif "AUTH" in error_code:
            reason = "There is an API authentication issue. Here is your cached summary:"
        elif "TIMEOUT" in error_code:
            reason = "The AI model took too long to respond. Here is your cached summary:"
        elif "NETWORK" in error_code:
            reason = "The AI service is temporarily unreachable. Here is your cached summary:"
        else:
            reason = "The AI service is momentarily unavailable. Here is your cached summary:"

        income = profile.get("total_income", 0)
        expenses = profile.get("total_expenses", 0)
        net = profile.get("net_savings", 0)
        avg_spend = profile.get("avg_monthly_spend", 0)
        top_cats = profile.get("top_expense_categories", [])
        top_cat = top_cats[0]["category"] if top_cats else "N/A"

        fallback_text = (
            f"{reason}\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Total Income | ${income:,.2f} |\n"
            f"| Total Expenses | ${expenses:,.2f} |\n"
            f"| Net Savings | ${net:,.2f} |\n"
            f"| Avg Monthly Spend | ${avg_spend:,.2f} |\n"
            f"| Top Spending Category | {top_cat} |"
        )

        guardrail_flags.append("LLM_UNAVAILABLE")

        audit.log_request(
            user_id=user_id,
            prompt=prompt,
            response_length=len(fallback_text),
            latency_ms=latency_ms,
            cache_hit=state.get("cache_hit", False),
            guardrail_flags=guardrail_flags,
            model_used="none",
        )

        return {
            "llm_response": fallback_text,
            "guardrail_flags": guardrail_flags,
            "latency_ms": latency_ms,
            "error": None,
        }

    return graceful_degradation
