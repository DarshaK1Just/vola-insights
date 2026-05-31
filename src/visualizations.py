"""
Visualization engine.

Required charts (per assessment spec):
  plot_monthly_spending_trend  — line chart + 3-month rolling average overlay
  plot_category_breakdown      — donut chart with TOTAL SPEND IN CENTER
  plot_income_vs_expense       — grouped bars (GREEN income / RED expense) + net line

Dynamic chart:
  generate_dynamic_chart       — LLM-driven flexible chart with any chart_type and
                                 any data_source ("custom" | named DataFrame-source)

All tools are exposed as OpenAI-compatible function schemas with user_id as first param
so the LLM can autonomously decide which to call and with which parameters.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _expenses(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["transaction_amount"] > 0]


def _income_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["transaction_amount"] < 0]


def _main_category(cat: str) -> str:
    if pd.isna(cat) or not str(cat).strip():
        return "Other"
    return str(cat).split(">")[0].strip()


def _sub_category(cat: str) -> str:
    """Return the second-level category (first subcategory). e.g. 'Food > Restaurants > Fast Food' → 'Restaurants'."""
    if pd.isna(cat) or not str(cat).strip():
        return "Other"
    parts = [p.strip() for p in str(cat).split(">")]
    return parts[1] if len(parts) > 1 else parts[0]


def _safe_fname(user_id: str, suffix: str) -> Path:
    uid = re.sub(r"[^a-z0-9_]", "_", user_id.lower())
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(Config.OUTPUT_DIR) / f"{uid}_{suffix}_{ts}.png"


def _filter_months(df: pd.DataFrame, months: int) -> pd.DataFrame:
    """Filter to last N months; fall back to full dataset if window is empty."""
    cutoff   = datetime.now() - timedelta(days=months * 30)
    filtered = df[df["transaction_date"] >= cutoff]
    if filtered.empty and not df.empty:
        logger.info("No data in last %d months; falling back to full dataset", months)
        return df.copy()
    return filtered


def _get_colors(scheme: str, n: int) -> list:
    palettes = {
        "financial": ["#4F8EF7","#10B981","#F59E0B","#EF4444","#8B5CF6","#EC4899","#14B8A6","#64748B","#F97316","#06B6D4"],
        "green":     ["#10B981","#34D399","#6EE7B7","#A7F3D0","#D1FAE5"],
        "blue":      ["#4F8EF7","#60A5FA","#93C5FD","#BFDBFE","#DBEAFE"],
        "red":       ["#EF4444","#F87171","#FCA5A5","#FECACA","#FEE2E2"],
        "gradient":  ["#4F8EF7","#7C3AED","#EC4899","#EF4444","#F59E0B"],
    }
    base = palettes.get(scheme, palettes["financial"])
    return (base * ((n // len(base)) + 1))[:n]


# ── Data-source resolver (for generate_dynamic_chart) ────────────────────────

def _resolve_data_source(
    user_df: pd.DataFrame,
    data_source: str,
    months: int = 6,
    top_n: int = 8,
    parent_category: str = "",
) -> list:
    """
    Fetch data from the user's DataFrame for a named data_source.
    Returns a list of {label, value} dicts ready for charting.
    """
    df = _filter_months(user_df, months)

    if data_source == "monthly_expenses":
        df2 = _expenses(df).copy()
        if df2.empty:
            return []
        df2["month"] = df2["transaction_date"].dt.to_period("M")
        grp = df2.groupby("month")["transaction_amount"].sum()
        return [{"label": str(m), "value": round(float(v), 2)} for m, v in sorted(grp.items())]

    if data_source == "monthly_income":
        df2 = _income_rows(df).copy()
        if df2.empty:
            return []
        df2["month"] = df2["transaction_date"].dt.to_period("M")
        grp = df2.groupby("month")["transaction_amount"].apply(lambda x: round(abs(float(x.sum())), 2))
        return [{"label": str(m), "value": float(v)} for m, v in sorted(grp.items())]

    if data_source == "category_totals":
        df2 = _expenses(df).copy()
        if df2.empty:
            return []
        df2["main_cat"] = df2["transaction_category_detail"].apply(_main_category)
        grp = df2.groupby("main_cat")["transaction_amount"].sum().sort_values(ascending=False)
        return [{"label": c, "value": round(float(v), 2)} for c, v in grp.head(top_n).items()]

    if data_source == "merchant_totals":
        df2 = _expenses(df)
        if df2.empty:
            return []
        grp = df2.groupby("merchant_name")["transaction_amount"].sum().sort_values(ascending=False)
        return [{"label": m, "value": round(float(v), 2)} for m, v in grp.head(top_n).items()]

    if data_source == "savings_rate":
        df2 = df.copy()
        if df2.empty:
            return []
        df2["month"] = df2["transaction_date"].dt.to_period("M")
        result = []
        for period, grp in df2.groupby("month"):
            inc = abs(float(_income_rows(grp)["transaction_amount"].sum()))
            exp = float(_expenses(grp)["transaction_amount"].sum())
            if inc > 0:
                result.append({"label": str(period), "value": round((inc - exp) / inc * 100, 1)})
        return sorted(result, key=lambda x: x["label"])

    if data_source == "monthly_net_savings":
        df2 = df.copy()
        if df2.empty:
            return []
        df2["month"] = df2["transaction_date"].dt.to_period("M")
        result = []
        for period, grp in df2.groupby("month"):
            inc = abs(float(_income_rows(grp)["transaction_amount"].sum()))
            exp = float(_expenses(grp)["transaction_amount"].sum())
            result.append({"label": str(period), "value": round(inc - exp, 2)})
        return sorted(result, key=lambda x: x["label"])

    if data_source == "subcategory_totals":
        # Shows subcategories of a parent category, e.g. Food → Restaurants, Groceries, Fast Food
        # Required for: "Show me my food spending" → top_subcategories with parent_category=Food
        df2 = _expenses(df).copy()
        if df2.empty:
            return []
        if parent_category:
            df2 = df2[df2["transaction_category_detail"].str.contains(
                parent_category, case=False, na=False)]
        if df2.empty:
            return []
        df2["sub_cat"] = df2["transaction_category_detail"].apply(_sub_category)
        grp = df2.groupby("sub_cat")["transaction_amount"].sum().sort_values(ascending=False)
        total = float(grp.sum())
        return [
            {
                "label": cat,
                "value": round(float(amt), 2),
                "pct": round(100 * float(amt) / total, 1) if total else 0,
            }
            for cat, amt in grp.head(top_n).items()
        ]

    return []


# ── 1. Monthly spending trend ─────────────────────────────────────────────────

def plot_monthly_spending_trend(
    user_df: pd.DataFrame,
    user_id: str,
    user_name: str,
    months: int = 1,                    # spec default: 1
    category_filter: Optional[str] = None,
) -> str:
    """Line chart: monthly expense totals + 3-month rolling average overlay."""
    # Use at least 3 months of data for a meaningful trend line
    effective_months = max(months, 3)
    df = _filter_months(user_df, effective_months)
    df = _expenses(df).copy()
    if category_filter:
        df = df[df["transaction_category_detail"].str.contains(
            category_filter, case=False, na=False)]
    if df.empty:
        logger.warning("plot_monthly_spending_trend: no data for user %s", user_id)
        return ""

    df["month"] = df["transaction_date"].dt.to_period("M")
    monthly = df.groupby("month")["transaction_amount"].sum().reset_index()
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()
    monthly = monthly.sort_values("month_dt")
    monthly["rolling_avg"] = monthly["transaction_amount"].rolling(3, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")

    ax.plot(monthly["month_dt"], monthly["transaction_amount"],
            color="#4F8EF7", linewidth=2.5, marker="o", markersize=5,
            label="Monthly spend", zorder=3)
    ax.plot(monthly["month_dt"], monthly["rolling_avg"],
            color="#F59E0B", linewidth=2.0, linestyle="--",
            label="3-month rolling avg", zorder=3)
    ax.fill_between(monthly["month_dt"], monthly["transaction_amount"],
                    alpha=0.08, color="#4F8EF7")

    title = f"{user_name} — Monthly Spending Trend"
    if category_filter:
        title += f" ({category_filter})"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Amount ($)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()

    path = _safe_fname(user_id, "trend")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved trend chart: %s", path)
    return str(path)


# ── 2. Category breakdown — donut with total in center ───────────────────────

def plot_category_breakdown(
    user_df: pd.DataFrame,
    user_id: str,
    user_name: str,
    period: str = "last_3_months",
    top_n: int = 7,
) -> str:
    """Donut chart: category split, TOTAL SPEND displayed in center."""
    period_months = {
        "last_month": 1, "last_3_months": 3,
        "last_6_months": 6, "last_year": 12,
    }.get(period.lower(), 3)

    df = _filter_months(user_df, period_months)
    df = _expenses(df).copy()
    if df.empty:
        logger.warning("plot_category_breakdown: no data for user %s", user_id)
        return ""

    df["main_cat"] = df["transaction_category_detail"].apply(_main_category)
    grp   = df.groupby("main_cat")["transaction_amount"].sum().sort_values(ascending=False)
    total = float(grp.sum())

    top_cats  = grp.head(top_n)
    other_amt = grp.iloc[top_n:].sum() if len(grp) > top_n else 0.0
    if other_amt > 0:
        top_cats = pd.concat([top_cats, pd.Series({"Other": other_amt})])

    colors = _get_colors("financial", len(top_cats))

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")

    wedges, _, autotexts = ax.pie(
        top_cats.values, labels=None, colors=colors,
        autopct="%1.1f%%", pctdistance=0.82, startangle=90,
        wedgeprops={"width": 0.52, "edgecolor": "white", "linewidth": 2.5},
    )
    for at in autotexts:
        at.set_fontsize(8.5); at.set_color("white"); at.set_fontweight("bold")

    # Total spend in center (assessment requirement)
    ax.text(0,  0.12, "Total Spend", ha="center", va="center",
            fontsize=11, color="#64748B")
    ax.text(0, -0.16, f"${total:,.0f}", ha="center", va="center",
            fontsize=20, fontweight="bold", color="#1E293B")

    ax.legend(
        wedges,
        [f"{cat}  ${amt:,.0f}" for cat, amt in top_cats.items()],
        title="Category", loc="lower left",
        bbox_to_anchor=(-0.32, -0.08), fontsize=9, title_fontsize=9,
    )
    ax.set_title(
        f"{user_name} — Spending by Category ({period.replace('_', ' ').title()})",
        fontsize=14, fontweight="bold", pad=20,
    )
    plt.tight_layout()

    path = _safe_fname(user_id, "categories")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved category chart: %s", path)
    return str(path)


# ── 3. Income vs Expense — green/red bars + net line ────────────────────────

def plot_income_vs_expense(
    user_df: pd.DataFrame,
    user_id: str,
    user_name: str,
    months: int = 6,
    show_net_line: bool = True,
) -> str:
    """Grouped bars (GREEN=income, RED=expense) + optional net savings line."""
    df = _filter_months(user_df, months).copy()
    if df.empty:
        logger.warning("plot_income_vs_expense: no data for user %s", user_id)
        return ""

    df["month"] = df["transaction_date"].dt.to_period("M")
    rows = []
    for period, grp in df.groupby("month"):
        inc = round(abs(float(_income_rows(grp)["transaction_amount"].sum())), 2)
        exp = round(float(_expenses(grp)["transaction_amount"].sum()), 2)
        rows.append({"month": period.to_timestamp(), "income": inc,
                     "expenses": exp, "net": round(inc - exp, 2)})
    if not rows:
        return ""

    mdf = pd.DataFrame(rows).sort_values("month")
    x   = np.arange(len(mdf))
    w   = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")

    ax.bar(x - w/2, mdf["income"],   w, label="Income",   color="#10B981", alpha=0.88, zorder=3)
    ax.bar(x + w/2, mdf["expenses"], w, label="Expenses",  color="#EF4444", alpha=0.88, zorder=3)

    if show_net_line:
        ax.plot(x, mdf["net"], color="#4F8EF7", linewidth=2.5, marker="D",
                markersize=7, label="Net savings", zorder=4)
        ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--", zorder=2)

    ax.set_title(f"{user_name} — Income vs Expenses (Last {months} months)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.strftime("%b %Y") for m in mdf["month"]],
                       rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    path = _safe_fname(user_id, "income_expense")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved income/expense chart: %s", path)
    return str(path)


# ── 4. Dynamic chart (LLM-driven) ─────────────────────────────────────────────

def generate_dynamic_chart(
    user_df: pd.DataFrame,
    user_id: str,
    user_name: str,
    chart_type: str = "bar",
    title: str = "Financial Overview",
    data_source: str = "custom",
    data: Optional[list] = None,
    x_label: str = "",
    y_label: str = "Amount ($)",
    color_scheme: str = "financial",
    show_values: bool = True,
    months: int = 6,
    top_n: int = 8,
    parent_category: str = "",
) -> str:
    """
    Render any chart the LLM decides to produce.

    data_source options
    ───────────────────
    "custom"               — LLM provides data directly as [{label, value}]
    "monthly_expenses"     — monthly expense totals from DataFrame
    "monthly_income"       — monthly income totals from DataFrame
    "category_totals"      — top categories by spend
    "subcategory_totals"   — subcategories of a parent (use parent_category="Food")
    "merchant_totals"      — top merchants by spend
    "savings_rate"         — monthly savings % (income-expenses)/income
    "monthly_net_savings"  — monthly net savings amount

    chart_type options
    ──────────────────
    "bar"  "line"  "donut"  "pie"  "area"
    """
    # Resolve data from DataFrame source if not custom
    if data_source != "custom" or not data:
        data = _resolve_data_source(user_df, data_source, months, top_n, parent_category)

    if not data:
        logger.warning("generate_dynamic_chart: no data resolved (user=%s, source=%s)", user_id, data_source)
        return ""

    labels = [str(d.get("label", "")) for d in data]
    values = [float(d.get("value", 0)) for d in data]
    colors = _get_colors(color_scheme, len(data))

    is_circular = chart_type in ("pie", "donut")
    fig, ax = plt.subplots(figsize=(10 if not is_circular else 9,
                                    5 if not is_circular else 7))
    fig.patch.set_facecolor("white")
    if not is_circular:
        ax.set_facecolor("#F8FAFC")

    # ── Render ────────────────────────────────────────────────────────────────
    if chart_type == "bar":
        bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.88, zorder=3)
        if show_values:
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(abs(v) for v in values) * 0.01,
                    f"${val:,.0f}", ha="center", va="bottom", fontsize=8,
                )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    elif chart_type == "line":
        ax.plot(range(len(labels)), values, color=colors[0],
                linewidth=2.5, marker="o", markersize=5, zorder=3)
        ax.fill_between(range(len(labels)), values, alpha=0.10, color=colors[0])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    elif chart_type == "area":
        ax.fill_between(range(len(labels)), values, alpha=0.35, color=colors[0])
        ax.plot(range(len(labels)), values, color=colors[0], linewidth=2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    elif chart_type in ("pie", "donut"):
        width_kwarg = {"width": 0.52} if chart_type == "donut" else {}
        wedges, _, autotexts = ax.pie(
            values, labels=None, colors=colors[:len(values)],
            autopct="%1.1f%%", pctdistance=0.82, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2, **width_kwarg},
        )
        for at in autotexts:
            at.set_fontsize(8.5); at.set_color("white"); at.set_fontweight("bold")
        if chart_type == "donut":
            total = sum(v for v in values if v > 0)
            label_txt = y_label.replace("Amount ($)", "Total") or "Total"
            ax.text(0,  0.10, label_txt,        ha="center", fontsize=11, color="#64748B")
            ax.text(0, -0.16, f"${total:,.0f}", ha="center", fontsize=18,
                    fontweight="bold", color="#1E293B")
        ax.legend(wedges, [f"{lb}  ${vl:,.0f}" for lb, vl in zip(labels, values)],
                  title="Category", loc="lower left",
                  bbox_to_anchor=(-0.32, -0.08), fontsize=9)

    # ── Axes decoration ───────────────────────────────────────────────────────
    ax.set_title(f"{user_name} — {title}", fontsize=14, fontweight="bold", pad=12)
    if not is_circular:
        if x_label:
            ax.set_xlabel(x_label, fontsize=11)
        if y_label:
            ax.set_ylabel(y_label, fontsize=11)
            # Use % format for savings_rate
            if "%" in y_label or data_source == "savings_rate":
                ax.yaxis.set_major_formatter(mticker.PercentFormatter())
            else:
                ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--", zorder=2)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()

    suffix = f"dynamic_{chart_type}_{re.sub(r'[^a-z0-9]', '_', data_source.lower())}"
    path = _safe_fname(user_id, suffix)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dynamic chart (%s/%s): %s", chart_type, data_source, path)
    return str(path)


# ── Tool schemas (OpenAI-compatible) — user_id included per spec ──────────────

VIZ_TOOL_NAMES = {
    "plot_monthly_spending_trend",
    "plot_category_breakdown",
    "plot_income_vs_expense",
    "generate_dynamic_chart",
}


def get_viz_tool_schemas() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "plot_monthly_spending_trend",
                "description": (
                    "Generate a line chart showing monthly spending trend with 3-month rolling average. "
                    "Use for: 'spending trend', 'how spending changed over time', monthly patterns."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id":         {"type": "string", "description": "Target user ID"},
                        "months":          {"type": "integer", "default": 1,
                                           "description": "Lookback period in months (default 1)"},
                        "category_filter": {"type": "string",
                                           "description": "Optional: filter to one category, e.g. 'Food'"},
                    },
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "plot_category_breakdown",
                "description": (
                    "Generate a donut chart showing spending by category with total spend in center. "
                    "Use for: 'where is my money going', 'category breakdown', 'what do I spend most on'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "Target user ID"},
                        "period":  {"type": "string", "default": "last_3_months",
                                   "enum": ["last_month", "last_3_months", "last_6_months", "last_year"]},
                        "top_n":   {"type": "integer", "default": 7,
                                   "description": "Top N categories (rest grouped as Other)"},
                    },
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "plot_income_vs_expense",
                "description": (
                    "Generate grouped bar chart (GREEN=income, RED=expense) with net savings line. "
                    "Use for: 'am I saving', 'income vs expenses', 'financial health', savings rate."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id":       {"type": "string", "description": "Target user ID"},
                        "months":        {"type": "integer", "default": 6},
                        "show_net_line": {"type": "boolean", "default": True,
                                         "description": "Overlay net savings line"},
                    },
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_dynamic_chart",
                "description": (
                    "Generate a fully custom chart from any data source or LLM-provided data. "
                    "Use when none of the fixed charts fit the user's question, or as a 4th chart "
                    "in a full financial report. "
                    "data_source options: monthly_expenses | monthly_income | category_totals | "
                    "merchant_totals | savings_rate | monthly_net_savings | custom. "
                    "chart_type options: bar | line | area | donut | pie."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id":      {"type": "string", "description": "Target user ID"},
                        "chart_type":   {"type": "string", "default": "bar",
                                        "enum": ["bar", "line", "area", "donut", "pie"]},
                        "title":        {"type": "string", "description": "Chart title"},
                        "data_source":  {
                            "type": "string",
                            "default": "category_totals",
                            "enum": [
                                "monthly_expenses", "monthly_income",
                                "category_totals", "subcategory_totals",
                                "merchant_totals", "savings_rate",
                                "monthly_net_savings", "custom",
                            ],
                            "description": (
                                "Named data source. Use 'subcategory_totals' with "
                                "parent_category='Food' for food subcategory breakdown."
                            ),
                        },
                        "parent_category": {
                            "type": "string",
                            "description": (
                                "Filter subcategory_totals to a parent category, "
                                "e.g. 'Food', 'Transport', 'Shopping'. "
                                "Required when data_source='subcategory_totals'."
                            ),
                        },
                        "data": {
                            "type": "array",
                            "description": "Custom data points [{label, value}] — only used when data_source='custom'",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "value": {"type": "number"},
                                },
                            },
                        },
                        "x_label":      {"type": "string", "default": ""},
                        "y_label":      {"type": "string", "default": "Amount ($)"},
                        "color_scheme": {"type": "string", "default": "financial",
                                        "enum": ["financial", "green", "blue", "red", "gradient"]},
                        "show_values":  {"type": "boolean", "default": True},
                        "months":       {"type": "integer", "default": 6,
                                        "description": "Lookback for data_source-driven charts"},
                        "top_n":        {"type": "integer", "default": 8},
                    },
                    "required": ["user_id", "chart_type", "title", "data_source"],
                },
            },
        },
    ]


# ── Execution dispatcher ───────────────────────────────────────────────────────

VIZ_TOOL_REGISTRY = {
    "plot_monthly_spending_trend": plot_monthly_spending_trend,
    "plot_category_breakdown":     plot_category_breakdown,
    "plot_income_vs_expense":      plot_income_vs_expense,
    "generate_dynamic_chart":      generate_dynamic_chart,
}


def execute_viz_tool(
    name: str,
    args: dict,
    user_df: pd.DataFrame,
    user_id: str,       # always from state — NOT from LLM args (security)
    user_name: str,
) -> str:
    """Execute a visualization tool; return saved PNG path or '' on failure."""
    fn = VIZ_TOOL_REGISTRY.get(name)
    if fn is None:
        logger.warning("Unknown viz tool: %s", name)
        return ""
    # Strip user_id from args — we always use the state's user_id for security
    safe_args = {k: v for k, v in args.items() if k != "user_id"}
    try:
        result = fn(user_df=user_df, user_id=user_id, user_name=user_name, **safe_args)
        return result or ""
    except Exception as exc:
        logger.error("Viz tool %s failed: %s", name, exc)
        return ""


# ── Legacy VisualizationEngine wrapper ────────────────────────────────────────

class VisualizationEngine:
    """Thin wrapper kept for backward-compatible imports."""

    def __init__(self, output_dir: str = None):
        pass

    def get_tool_schemas(self) -> list:
        return get_viz_tool_schemas()

    def execute_tool(self, tool_name, tool_args, user_df, user_id, user_name) -> str:
        return execute_viz_tool(tool_name, tool_args, user_df, user_id, user_name)
