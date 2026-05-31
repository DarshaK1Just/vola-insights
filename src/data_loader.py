import logging
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "user_id",
    "user_name",
    "transaction_date",
    "transaction_amount",
    "transaction_category_detail",
    "merchant_name",
]


def load_transactions(filepath: str) -> pd.DataFrame:
    """Load and validate transaction data from an Excel file.

    Args:
        filepath: Absolute path to the .xlsx file.

    Returns:
        Validated DataFrame with transaction_date as datetime dtype.

    Raises:
        ValueError: If required columns are missing.
        FileNotFoundError: If the file does not exist.
    """
    logger.info("Loading transactions from %s", filepath)
    df = pd.read_excel(filepath)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    logger.info("Loaded %d rows from %s", len(df), filepath)
    return df


def parse_category(category_detail: str) -> tuple:
    """Parse a category_detail string into (sub_category, main_category).

    Args:
        category_detail: e.g. "RENT_HOUSING", "FASTFOOD_FOOD", "FREELANCE_INCOME"

    Returns:
        Tuple of (sub_category, main_category).
        Returns ("UNKNOWN", "UNKNOWN") for empty or malformed input.
    """
    if not category_detail or not isinstance(category_detail, str):
        return ("UNKNOWN", "UNKNOWN")

    cat = category_detail.strip()
    if not cat:
        return ("UNKNOWN", "UNKNOWN")

    parts = cat.split("_")
    main_cat = parts[-1]
    sub_cat = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
    return (sub_cat, main_cat)


def is_income(amount: float) -> bool:
    """Return True if the amount represents income (negative value).

    Args:
        amount: Transaction amount. Negative = income, positive = expense.

    Returns:
        True if amount < 0.
    """
    return amount < 0


def compute_user_profile(user_df: pd.DataFrame, user_id: str, user_name: str) -> dict:
    """Compute a comprehensive financial profile for a single user.

    IMPORTANT amount convention: NEGATIVE = INCOME, POSITIVE = EXPENSE.

    Args:
        user_df: DataFrame filtered to a single user's transactions.
        user_id: The user's identifier string.
        user_name: The user's display name.

    Returns:
        Dictionary with all financial summary keys.
    """
    logger.debug("Computing profile for user %s (%s)", user_id, user_name)

    # Date range
    date_range_start = user_df["transaction_date"].min().isoformat()
    date_range_end = user_df["transaction_date"].max().isoformat()

    # Split income vs expense rows
    income_mask = user_df["transaction_amount"] < 0
    expense_mask = user_df["transaction_amount"] > 0

    income_rows = user_df[income_mask]
    expense_rows = user_df[expense_mask]

    total_income = float(abs(income_rows["transaction_amount"].sum()))
    total_expenses = float(expense_rows["transaction_amount"].sum())
    net_savings = total_income - total_expenses

    # Number of distinct calendar months in the dataset
    months = user_df["transaction_date"].dt.to_period("M").unique()
    num_months = max(len(months), 1)

    avg_monthly_income = total_income / num_months
    avg_monthly_spend = total_expenses / num_months

    # Top 5 expense categories by main category
    expense_rows = expense_rows.copy()
    expense_rows["main_category"] = expense_rows["transaction_category_detail"].apply(
        lambda x: parse_category(x)[1]
    )
    category_totals = (
        expense_rows.groupby("main_category")["transaction_amount"]
        .sum()
        .sort_values(ascending=False)
    )
    top_5_expense_categories = [
        {"category": cat, "amount": round(float(amt), 2)}
        for cat, amt in category_totals.head(5).items()
    ]

    # Income breakdown by full category_detail
    income_rows = income_rows.copy()
    income_breakdown = (
        income_rows.groupby("transaction_category_detail")["transaction_amount"]
        .sum()
        .apply(lambda x: round(float(abs(x)), 2))
        .to_dict()
    )

    # Monthly summary keyed by "YYYY-MM"
    user_df = user_df.copy()
    user_df["month_key"] = user_df["transaction_date"].dt.to_period("M").astype(str)

    monthly_summary: dict = {}
    for month_key, month_df in user_df.groupby("month_key"):
        m_income = float(abs(month_df.loc[month_df["transaction_amount"] < 0, "transaction_amount"].sum()))
        m_expenses = float(month_df.loc[month_df["transaction_amount"] > 0, "transaction_amount"].sum())
        monthly_summary[month_key] = {
            "income": round(m_income, 2),
            "expenses": round(m_expenses, 2),
            "net": round(m_income - m_expenses, 2),
        }

    # Data quality flags
    data_quality_flags: list = []

    # Exact duplicates (same date + category + amount)
    exact_dup_cols = ["user_id", "transaction_date", "transaction_category_detail", "transaction_amount"]
    if user_df.duplicated(subset=exact_dup_cols).any():
        data_quality_flags.append("duplicate_transactions_detected")
        logger.warning("Exact duplicate transactions detected for user %s", user_id)

    # Multiple rent/housing payments on the same calendar day (different amounts = suspected double-billing)
    housing_rows = user_df[user_df["transaction_category_detail"].str.contains("HOUSING", na=False)]
    if housing_rows.groupby("transaction_date").size().gt(1).any():
        data_quality_flags.append("multiple_housing_charges_same_day")
        logger.warning("Multiple housing charges on same day detected for user %s", user_id)

    # More than 2 rent payments in any single month (suspicious)
    if not housing_rows.empty:
        housing_rows = housing_rows.copy()
        housing_rows["month_key"] = housing_rows["transaction_date"].dt.to_period("M")
        if housing_rows.groupby("month_key").size().gt(2).any():
            if "multiple_housing_charges_same_day" not in data_quality_flags:
                data_quality_flags.append("multiple_housing_charges_per_month")
            logger.warning("Multiple housing charges in same month for user %s", user_id)

    # top_expense_categories: normalised key expected by nodes.py / analysis_tools.py
    top_expense_categories = [
        {"category": c["category"], "total": c["amount"]}
        for c in top_5_expense_categories
    ]

    return {
        "user_id": user_id,
        "user_name": user_name,
        # Normalised keys used by nodes.py
        "date_range": {
            "start": date_range_start[:10] if date_range_start else "",
            "end": date_range_end[:10] if date_range_end else "",
        },
        "total_transactions": len(user_df),
        "top_expense_categories": top_expense_categories,
        # Legacy keys kept for backward compat
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "transaction_count": len(user_df),
        "top_5_expense_categories": top_5_expense_categories,
        # Core financial figures
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "net_savings": round(net_savings, 2),
        "avg_monthly_income": round(avg_monthly_income, 2),
        "avg_monthly_spend": round(avg_monthly_spend, 2),
        "income_breakdown": income_breakdown,
        "monthly_summary": monthly_summary,
        "data_quality_flags": data_quality_flags,
    }


def get_user_data(df: pd.DataFrame, user_id: str) -> tuple:
    """Filter the full DataFrame to a single user.

    Args:
        df: Full transactions DataFrame.
        user_id: The user identifier to filter by.

    Returns:
        Tuple of (filtered_df, user_name).

    Raises:
        ValueError: If the user_id is not found in the DataFrame.
    """
    user_df = df[df["user_id"] == user_id].copy()
    if user_df.empty:
        raise ValueError(f"User not found: {user_id}")

    user_name = str(user_df["user_name"].iloc[0])
    logger.debug("Retrieved %d rows for user %s (%s)", len(user_df), user_id, user_name)
    return (user_df, user_name)
