"""
DataFrame-first analysis tools.

Each function executes pandas operations on the user's filtered DataFrame and
returns a JSON-serialisable dict.  These are exposed to the LLM as OpenAI-format
function schemas so the model can call them to answer the user's query.

The tool name + args are stored in query_history as "pandas_operation" per spec:
  user:{id}:query_history -> [{prompt, pandas_operation, result_summary}]
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Period helpers ────────────────────────────────────────────────────────────

def _parse_period(period: str, anchor: datetime):
    """Return (start, end) datetime for a named period string."""
    p = period.lower().replace("-", "_").replace(" ", "_")
    end = anchor
    if p in ("last_month", "last_1_month"):
        first_this = anchor.replace(day=1)
        end = first_this - timedelta(days=1)
        start = end.replace(day=1)
    elif p == "last_3_months":
        start = (anchor - timedelta(days=92)).replace(day=1)
    elif p == "last_6_months":
        start = (anchor - timedelta(days=183)).replace(day=1)
    elif p == "last_year":
        start = anchor.replace(year=anchor.year - 1, month=1, day=1)
    elif p == "this_month":
        start = anchor.replace(day=1)
    elif p == "this_year":
        start = anchor.replace(month=1, day=1)
    else:
        start = (anchor - timedelta(days=92)).replace(day=1)
    return start.replace(hour=0, minute=0, second=0, microsecond=0), end


def _filter_months(df: pd.DataFrame, months: int) -> pd.DataFrame:
    cutoff = datetime.now() - timedelta(days=months * 30)
    return df[df["transaction_date"] >= cutoff]


def _expenses(df: pd.DataFrame) -> pd.DataFrame:
    """Positive amounts = expenses per schema convention."""
    return df[df["transaction_amount"] > 0]


def _income_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Negative amounts = income per schema convention."""
    return df[df["transaction_amount"] < 0]


def _main_category(cat: str) -> str:
    if pd.isna(cat) or not str(cat).strip():
        return "Other"
    parts = str(cat).split(">")
    return parts[0].strip() if parts else "Other"


# ── Tool implementations ──────────────────────────────────────────────────────

def _filter_or_all(user_df: pd.DataFrame, months: int) -> tuple:
    """Filter by months; fall back to full dataset if window is empty.
    Returns (filtered_df, effective_months_note)."""
    df = _filter_months(user_df, months)
    if not df.empty:
        return df, ""
    # Fallback: use all available data
    if not user_df.empty:
        start = user_df["transaction_date"].min()
        end = user_df["transaction_date"].max()
        note = (f" (Note: no transactions in the last {months} month(s); "
                f"showing full history from {start.date()} to {end.date()})")
        return user_df.copy(), note
    return user_df.copy(), ""


def get_spending_by_category(
    user_df: pd.DataFrame,
    months: int = 3,
    top_n: int = 10,
) -> dict:
    """Pandas: groupby main_category on expense rows for the last N months."""
    df, note = _filter_or_all(user_df, months)
    df = _expenses(df).copy()
    if df.empty:
        return {"total_expenses": 0, "categories": [], "period_months": months,
                "message": f"No expense transactions found in the last {months} month(s)."}
    df["main_cat"] = df["transaction_category_detail"].apply(_main_category)
    grp = df.groupby("main_cat")["transaction_amount"].sum().sort_values(ascending=False)
    total = float(grp.sum())
    cats = [
        {"category": cat, "amount": round(float(amt), 2),
         "pct": round(100 * float(amt) / total, 1) if total else 0}
        for cat, amt in grp.head(top_n).items()
    ]
    result = {
        "total_expenses": round(total, 2),
        "categories": cats,
        "period_months": months,
        "transaction_count": len(df),
    }
    if note:
        result["note"] = note
    return result


def get_merchant_analysis(
    user_df: pd.DataFrame,
    months: int = 3,
    top_n: int = 10,
) -> dict:
    """Pandas: groupby merchant_name on expense rows for the last N months."""
    df, note = _filter_or_all(user_df, months)
    df = _expenses(df)
    if df.empty:
        return {"merchants": [], "period_months": months,
                "message": f"No transactions found in the last {months} month(s)."}
    grp = (
        df.groupby("merchant_name")["transaction_amount"]
        .agg(total="sum", count="count")
        .sort_values("total", ascending=False)
        .head(top_n)
        .reset_index()
    )
    total = float(df["transaction_amount"].sum())
    merchants = [
        {
            "merchant": row["merchant_name"],
            "total": round(float(row["total"]), 2),
            "count": int(row["count"]),
            "pct": round(100 * float(row["total"]) / total, 1) if total else 0,
        }
        for _, row in grp.iterrows()
    ]
    return {"merchants": merchants, "period_months": months}


def get_monthly_trend(user_df: pd.DataFrame, months: int = 12) -> dict:
    """Pandas: resample by month, compute income/expense/net per month."""
    df = _filter_months(user_df, months).copy()
    if df.empty:
        return {"months": [], "avg_monthly_expense": 0,
                "message": f"No transactions found in the last {months} month(s)."}
    df["month"] = df["transaction_date"].dt.to_period("M")
    result = []
    for period, grp in df.groupby("month"):
        inc = round(abs(float(_income_rows(grp)["transaction_amount"].sum())), 2)
        exp = round(float(_expenses(grp)["transaction_amount"].sum()), 2)
        result.append({
            "month": str(period),
            "income": inc,
            "expenses": exp,
            "net": round(inc - exp, 2),
        })
    result.sort(key=lambda x: x["month"])
    expenses_list = [r["expenses"] for r in result]
    avg_exp = round(float(np.mean(expenses_list)), 2) if expenses_list else 0
    return {"months": result, "avg_monthly_expense": avg_exp}


def get_income_analysis(user_df: pd.DataFrame, months: int = 6) -> dict:
    """Pandas: groupby main_category on income rows for the last N months."""
    df = _filter_months(user_df, months)
    df = _income_rows(df).copy()
    if df.empty:
        return {"total_income": 0, "sources": [], "period_months": months,
                "message": f"No income transactions found in the last {months} month(s)."}
    df["amt_pos"] = df["transaction_amount"].abs()
    df["main_cat"] = df["transaction_category_detail"].apply(_main_category)
    grp = df.groupby("main_cat")["amt_pos"].sum().sort_values(ascending=False)
    total = float(grp.sum())
    sources = [
        {"source": cat, "amount": round(float(amt), 2),
         "pct": round(100 * float(amt) / total, 1) if total else 0}
        for cat, amt in grp.items()
    ]
    return {"total_income": round(total, 2), "sources": sources, "period_months": months}


def get_period_stats(user_df: pd.DataFrame, period: str = "last_month") -> dict:
    """Pandas: filter by named period, compute income/expense/net summary stats."""
    anchor = datetime.now()
    start, end = _parse_period(period, anchor)
    df = user_df[
        (user_df["transaction_date"] >= start)
        & (user_df["transaction_date"] <= end)
    ]
    if df.empty:
        return {
            "period": period,
            "message": f"No transactions found for {period} ({start.date()} to {end.date()}).",
        }
    inc = round(abs(float(_income_rows(df)["transaction_amount"].sum())), 2)
    exp = round(float(_expenses(df)["transaction_amount"].sum()), 2)
    # Top expense category for this period
    exp_df = _expenses(df).copy()
    top_cat = ""
    if not exp_df.empty:
        exp_df["main_cat"] = exp_df["transaction_category_detail"].apply(_main_category)
        top_cat = exp_df.groupby("main_cat")["transaction_amount"].sum().idxmax()
    return {
        "period": period,
        "start": str(start.date()),
        "end": str(end.date()),
        "total_income": inc,
        "total_expenses": exp,
        "net_savings": round(inc - exp, 2),
        "transaction_count": len(df),
        "top_expense_category": top_cat,
    }


def compare_periods(
    user_df: pd.DataFrame,
    period1: str = "last_month",
    period2: str = "last_3_months",
) -> dict:
    """Pandas: compare expense totals between two named periods."""
    s1 = get_period_stats(user_df, period1)
    s2 = get_period_stats(user_df, period2)
    if "message" in s1 or "message" in s2:
        return {"error": "Insufficient data for one or both periods.", "period1": s1, "period2": s2}
    e1, e2 = s1["total_expenses"], s2["total_expenses"]
    change_pct = round(100 * (e1 - e2) / e2, 1) if e2 else 0
    return {
        "period1": s1,
        "period2": s2,
        "expense_change_pct": change_pct,
        "trend": "higher" if change_pct > 0 else "lower",
    }


# ── Tool registry & schemas ───────────────────────────────────────────────────

ANALYSIS_TOOL_REGISTRY = {
    "get_spending_by_category": get_spending_by_category,
    "get_merchant_analysis": get_merchant_analysis,
    "get_monthly_trend": get_monthly_trend,
    "get_income_analysis": get_income_analysis,
    "get_period_stats": get_period_stats,
    "compare_periods": compare_periods,
}

ANALYSIS_TOOL_NAMES = set(ANALYSIS_TOOL_REGISTRY.keys())


def get_analysis_tool_schemas() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_spending_by_category",
                "description": (
                    "Get total spending broken down by category for the last N months. "
                    "Use when the user asks about categories, what they spend most on, "
                    "or where their money goes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "months": {"type": "integer", "default": 3, "description": "Lookback months"},
                        "top_n": {"type": "integer", "default": 10, "description": "Top N categories"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_merchant_analysis",
                "description": (
                    "Get top merchants by spend for the last N months. "
                    "Use when the user asks about specific stores, merchants, or suppliers."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "months": {"type": "integer", "default": 3},
                        "top_n": {"type": "integer", "default": 10},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_monthly_trend",
                "description": (
                    "Get monthly income vs expense trend for the last N months. "
                    "Use when the user asks about trends, spending over time, or monthly patterns."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "months": {"type": "integer", "default": 12},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_income_analysis",
                "description": (
                    "Get income breakdown by source for the last N months. "
                    "Use when the user asks about income, salary, or earnings."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "months": {"type": "integer", "default": 6},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_period_stats",
                "description": (
                    "Get summary stats (income, expenses, net savings) for a named period. "
                    "Use for specific time period questions like 'last month', 'this year'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "period": {
                            "type": "string",
                            "default": "last_month",
                            "enum": [
                                "last_month", "last_3_months", "last_6_months",
                                "last_year", "this_month", "this_year",
                            ],
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_periods",
                "description": (
                    "Compare spending between two time periods. "
                    "Use when the user asks about change, improvement, or comparison."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "period1": {"type": "string", "default": "last_month"},
                        "period2": {"type": "string", "default": "last_3_months"},
                    },
                },
            },
        },
    ]


def execute_analysis_tool(name: str, args: dict, user_df: pd.DataFrame) -> dict:
    """Execute a named analysis tool with given args on the user's DataFrame."""
    fn = ANALYSIS_TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown analysis tool: {name}"}
    try:
        return fn(user_df, **args)
    except Exception as exc:
        logger.error("Analysis tool %s failed: %s", name, exc)
        return {"error": str(exc)}


def format_tool_result_for_llm(name: str, result: dict) -> str:
    """Format a tool result as compact JSON string for the LLM synthesis step."""
    try:
        return json.dumps(result, indent=2, default=str)
    except Exception:
        return str(result)


def summarise_tool_result(name: str, result: dict) -> str:
    """One-line summary stored in query_history as result_summary."""
    if "error" in result:
        return f"{name}: error — {result['error']}"
    if name == "get_spending_by_category":
        cats = result.get("categories", [])
        top = cats[0] if cats else {}
        return (
            f"Top category: {top.get('category', '?')} "
            f"${top.get('amount', 0):.0f} ({top.get('pct', 0):.0f}%), "
            f"Total expenses: ${result.get('total_expenses', 0):.0f}"
        )
    if name == "get_monthly_trend":
        months = result.get("months", [])
        if months:
            last = months[-1]
            return (
                f"Last recorded month: {last.get('month', '?')}, "
                f"expenses ${last.get('expenses', 0):.0f}, net ${last.get('net', 0):.0f}"
            )
        return "No monthly data"
    if name == "get_period_stats":
        return (
            f"Period {result.get('period', '?')}: "
            f"income ${result.get('total_income', 0):.0f}, "
            f"expenses ${result.get('total_expenses', 0):.0f}, "
            f"net ${result.get('net_savings', 0):.0f}"
        )
    if name == "get_income_analysis":
        return f"Total income: ${result.get('total_income', 0):.0f}"
    if name == "compare_periods":
        return (
            f"Expense change: {result.get('expense_change_pct', 0):+.1f}% "
            f"({result.get('trend', '?')})"
        )
    if name == "get_merchant_analysis":
        ms = result.get("merchants", [])
        top = ms[0] if ms else {}
        return f"Top merchant: {top.get('merchant', '?')} ${top.get('total', 0):.0f}"
    return str(result)[:200]
