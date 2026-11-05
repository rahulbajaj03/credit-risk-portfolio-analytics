"""Portfolio analytics / profiling.

Produces the cross-tabs and segment summaries that back both the dashboard
portfolio view and the top-3 insights in the solution overview. All numbers
come straight from account_master so the LLM feature can quote them safely.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from . import config


def load_account_master() -> pd.DataFrame:
    return pd.read_parquet(config.ACCOUNT_DATASET)


def default_rate_by(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Default rate + exposure by a single dimension, sorted worst-first."""
    g = (
        df.groupby(dim, observed=True)
        .agg(
            n_accounts=("account_id", "count"),
            default_rate=(config.TARGET, "mean"),
            total_exposure=("loan_amount", "sum"),
        )
        .reset_index()
        .sort_values("default_rate", ascending=False)
    )
    g["default_rate"] = g["default_rate"].round(4)
    return g


def crosstab_default(df: pd.DataFrame, row: str, col: str) -> pd.DataFrame:
    """Cross-dimensional default-rate table (row x col)."""
    return (
        pd.crosstab(df[row], df[col], values=df[config.TARGET], aggfunc="mean")
        .round(4)
    )


def portfolio_summary(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Bundle the standard profiling cuts used across the app."""
    return {
        "by_loan_type": default_rate_by(df, "loan_type"),
        "by_employment": default_rate_by(df, "employment_type"),
        "by_region": default_rate_by(df, "region"),
        "by_income_band": default_rate_by(df, "income_band"),
        "by_age_band": default_rate_by(df, "age_band"),
        "employment_x_income": crosstab_default(df, "employment_type", "income_band"),
        "region_x_loan_type": crosstab_default(df, "region", "loan_type"),
    }


def headline_kpis(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "n_accounts": int(len(df)),
        "n_customers": int(df["customer_id"].nunique()),
        "portfolio_default_rate": float(df[config.TARGET].mean()),
        "total_exposure": float(df["loan_amount"].sum()),
        "defaulted_exposure": float(df.loc[df[config.TARGET] == 1, "loan_amount"].sum()),
        "pct_with_bureau_writeoff": float((df["bureau_n_written_off"] > 0).mean()),
    }


if __name__ == "__main__":
    df = load_account_master()
    print("KPIs:", headline_kpis(df))
    for name, tbl in portfolio_summary(df).items():
        print(f"\n=== {name} ===")
        print(tbl)
