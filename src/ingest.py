"""Data ingestion and cleaning.

Loads the 5 raw tables, resolves the known (intentional) data-quality issues,
engineers bureau / repayment / collections aggregates, and joins everything
into two analysis-ready datasets:

    * account_master   - 1 row per loan account (holds the modelling target)
    * customer_master  - 1 row per customer (for profiling / segmentation)

Every data-quality decision is logged to a JSON report so it can be audited
and explained in the interview.
"""
from __future__ import annotations

import json
from typing import Dict

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_raw() -> Dict[str, pd.DataFrame]:
    """Read the 5 CSVs. Date columns are parsed where present."""
    customers = pd.read_csv(config.RAW_FILES["customers"], parse_dates=["customer_since"])
    loans = pd.read_csv(config.RAW_FILES["loans"], parse_dates=["origination_date"])
    bureau = pd.read_csv(
        config.RAW_FILES["bureau"], parse_dates=["date_opened", "date_closed"]
    )
    repayment = pd.read_csv(config.RAW_FILES["repayment"])
    collections = pd.read_csv(config.RAW_FILES["collections"], parse_dates=["event_date"])
    return {
        "customers": customers,
        "loans": loans,
        "bureau": bureau,
        "repayment": repayment,
        "collections": collections,
    }


# ---------------------------------------------------------------------------
# Clean customers
# ---------------------------------------------------------------------------
def clean_customers(customers: pd.DataFrame, dq: dict) -> pd.DataFrame:
    df = customers.copy()

    # 1. Normalise inconsistent gender labels ("M"/"F" -> "Male"/"Female").
    before = df["gender"].value_counts().to_dict()
    df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})
    dq["gender_normalisation"] = {
        "decision": "Mapped 'M'->'Male' and 'F'->'Female' so the 4 labels collapse to 2.",
        "labels_before": {k: int(v) for k, v in before.items()},
        "labels_after": {k: int(v) for k, v in df["gender"].value_counts().items()},
    }

    # 2. Missing numeric profile fields: years_employed, num_dependents,
    #    years_at_residence. We impute with the median and keep an explicit
    #    "_was_missing" flag rather than silently dropping rows, because the
    #    fact that a field is missing can itself be predictive.
    for col in ["years_employed", "num_dependents", "years_at_residence"]:
        n_missing = int(df[col].isna().sum())
        median = float(df[col].median())
        df[f"{col}_was_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(median)
        dq[f"impute_{col}"] = {
            "decision": f"Median-imputed {n_missing} missing values with {median:.2f}; "
            f"kept {col}_was_missing flag.",
            "n_missing": n_missing,
            "median": median,
        }

    # 3. Derived: income band for segmentation.
    df["income_band"] = pd.cut(
        df["annual_income"],
        bins=[0, 25_000, 50_000, 100_000, 200_000, np.inf],
        labels=["<25k", "25-50k", "50-100k", "100-200k", "200k+"],
    )
    # 4. Derived: age band.
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, np.inf],
        labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
    )
    return df


# ---------------------------------------------------------------------------
# Clean loans
# ---------------------------------------------------------------------------
def clean_loans(loans: pd.DataFrame, dq: dict) -> pd.DataFrame:
    df = loans.copy()

    # Missing interest_rate_pct. Secured and unsecured loans price very
    # differently, so impute by loan_type median rather than a global median.
    n_missing = int(df["interest_rate_pct"].isna().sum())
    df["interest_rate_was_missing"] = df["interest_rate_pct"].isna().astype(int)
    df["interest_rate_pct"] = df.groupby("loan_type")["interest_rate_pct"].transform(
        lambda s: s.fillna(s.median())
    )
    dq["impute_interest_rate_pct"] = {
        "decision": f"Imputed {n_missing} missing interest rates with the per-loan_type "
        "median (secured vs unsecured price differently); kept missing flag.",
        "n_missing": n_missing,
    }
    return df


# ---------------------------------------------------------------------------
# Feature engineering: bureau (many-to-one with customers)
# ---------------------------------------------------------------------------
def build_bureau_features(bureau: pd.DataFrame, dq: dict) -> pd.DataFrame:
    df = bureau.copy()

    # Missing days_past_due in bureau: treat as 0 (last report showed no DPD).
    n_missing = int(df["days_past_due"].isna().sum())
    df["days_past_due"] = df["days_past_due"].fillna(0)
    dq["bureau_days_past_due"] = {
        "decision": f"Filled {n_missing} missing bureau DPD with 0 "
        "(interpreted as no delinquency reported at last update).",
        "n_missing": n_missing,
    }

    df["is_written_off"] = (df["account_status"] == "Written-Off").astype(int)
    df["is_active"] = (df["account_status"] == "Active").astype(int)
    df["utilisation"] = (
        df["outstanding_amount"] / df["sanctioned_amount"].replace(0, np.nan)
    ).clip(0, 5)

    agg = df.groupby("customer_id").agg(
        bureau_n_lines=("bureau_id", "count"),
        bureau_n_written_off=("is_written_off", "sum"),
        bureau_n_active=("is_active", "sum"),
        bureau_total_sanctioned=("sanctioned_amount", "sum"),
        bureau_total_outstanding=("outstanding_amount", "sum"),
        bureau_max_dpd=("days_past_due", "max"),
        bureau_mean_dpd=("days_past_due", "mean"),
        bureau_mean_utilisation=("utilisation", "mean"),
    )
    agg["bureau_writeoff_ratio"] = agg["bureau_n_written_off"] / agg["bureau_n_lines"]
    agg["bureau_overall_utilisation"] = (
        agg["bureau_total_outstanding"] / agg["bureau_total_sanctioned"].replace(0, np.nan)
    ).clip(0, 5)
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Feature engineering: repayment history (many-to-one with accounts)
# ---------------------------------------------------------------------------
def build_repayment_features(repayment: pd.DataFrame) -> pd.DataFrame:
    df = repayment.copy()
    df["is_missed"] = (df["payment_status"] == "Missed").astype(int)
    df["is_late"] = (df["payment_status"] == "Late").astype(int)
    df["shortfall"] = (df["emi_due"] - df["emi_paid"]).clip(lower=0)

    agg = df.groupby("account_id").agg(
        repay_n_months=("payment_month", "count"),
        repay_n_missed=("is_missed", "sum"),
        repay_n_late=("is_late", "sum"),
        repay_max_dpd=("days_past_due", "max"),
        repay_mean_dpd=("days_past_due", "mean"),
        repay_total_shortfall=("shortfall", "sum"),
        repay_last_outstanding=("outstanding_balance", "last"),
    )
    agg["repay_missed_ratio"] = agg["repay_n_missed"] / agg["repay_n_months"]
    agg["repay_late_ratio"] = agg["repay_n_late"] / agg["repay_n_months"]

    # Recent-behaviour signal: missed payments in the last 6 reported months.
    last6 = (
        df.sort_values(["account_id", "payment_month"])
        .groupby("account_id")
        .tail(6)
        .groupby("account_id")
        .agg(repay_recent_missed=("is_missed", "sum"), repay_recent_dpd=("days_past_due", "max"))
    )
    agg = agg.join(last6)
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Feature engineering: collections (sparse - only delinquent accounts)
# ---------------------------------------------------------------------------
def build_collections_features(collections: pd.DataFrame) -> pd.DataFrame:
    df = collections.copy()
    df["is_legal"] = df["action_type"].isin(["Legal Notice", "Written-Off Referral"]).astype(int)
    df["is_broken_promise"] = (df["outcome"] == "No Response").astype(int)

    agg = df.groupby("account_id").agg(
        coll_n_events=("collection_event_id", "count"),
        coll_total_recovered=("amount_recovered", "sum"),
        coll_n_legal=("is_legal", "sum"),
        coll_n_no_response=("is_broken_promise", "sum"),
    )
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Join everything
# ---------------------------------------------------------------------------
def build_datasets(save: bool = True) -> Dict[str, pd.DataFrame]:
    dq: dict = {}
    raw = load_raw()

    customers = clean_customers(raw["customers"], dq)
    loans = clean_loans(raw["loans"], dq)
    bureau_feat = build_bureau_features(raw["bureau"], dq)
    repay_feat = build_repayment_features(raw["repayment"])
    coll_feat = build_collections_features(raw["collections"])

    # Account master: loans is the spine (holds the target).
    acct = loans.merge(customers, on="customer_id", how="left")
    acct = acct.merge(bureau_feat, on="customer_id", how="left")
    acct = acct.merge(repay_feat, on="account_id", how="left")
    acct = acct.merge(coll_feat, on="account_id", how="left")

    # Accounts with no collections activity simply never went to collections.
    for c in ["coll_n_events", "coll_total_recovered", "coll_n_legal", "coll_n_no_response"]:
        acct[c] = acct[c].fillna(0)
    acct["has_collections"] = (acct["coll_n_events"] > 0).astype(int)

    # Bureau features may be missing if a customer has no external lines.
    bureau_cols = [c for c in acct.columns if c.startswith("bureau_")]
    acct[bureau_cols] = acct[bureau_cols].fillna(0)

    # Loan tenure in years for readability.
    ref = pd.Timestamp(config.REFERENCE_DATE)
    acct["loan_age_months"] = (
        (ref - acct["origination_date"]).dt.days / 30.44
    ).round(1)

    dq["join_logic"] = {
        "spine": "loan_accounts (grain = account_id, holds target_default)",
        "joins": [
            "customers on customer_id (left) - demographic dimension",
            "bureau aggregates on customer_id (left) - external credit summary",
            "repayment aggregates on account_id (left) - monthly behaviour",
            "collections aggregates on account_id (left) - recovery activity",
        ],
        "n_accounts": int(len(acct)),
        "n_customers": int(acct["customer_id"].nunique()),
        "default_rate": float(acct[config.TARGET].mean()),
    }

    # Customer master: for profiling. Aggregate account-level facts up to the
    # customer, keeping demographic columns.
    cust_loan_agg = (
        loans.groupby("customer_id")
        .agg(
            n_accounts=("account_id", "count"),
            total_loan_amount=("loan_amount", "sum"),
            any_default=(config.TARGET, "max"),
            default_rate=(config.TARGET, "mean"),
        )
        .reset_index()
    )
    cust = customers.merge(cust_loan_agg, on="customer_id", how="left")
    cust = cust.merge(bureau_feat, on="customer_id", how="left")
    cust[bureau_cols] = cust[bureau_cols].fillna(0)

    if save:
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        acct.to_parquet(config.ACCOUNT_DATASET, index=False)
        cust.to_parquet(config.CUSTOMER_DATASET, index=False)
        with open(config.DQ_REPORT, "w") as f:
            json.dump(dq, f, indent=2, default=str)

    return {"account_master": acct, "customer_master": cust, "dq_report": dq}


if __name__ == "__main__":
    out = build_datasets(save=True)
    acct = out["account_master"]
    print(f"account_master: {acct.shape}")
    print(f"customer_master: {out['customer_master'].shape}")
    print(f"default rate: {acct[config.TARGET].mean():.3f}")
    print("DQ report written to", config.DQ_REPORT)
