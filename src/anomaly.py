"""Anomaly detection (AI feature 2).

Flags accounts whose repayment / bureau behaviour is unusual compared with
their *peer segment* (same loan_type x income_band). We combine two signals:

    1. Peer z-scores on key behavioural metrics (interpretable, rule-like).
    2. IsolationForest multivariate outlier score (catches odd combinations).

Each flagged account carries a plain-English reason list so the analyst can
see *why* it was flagged and record a confirm / dismiss decision.

Trigger:         composite anomaly score above the 95th percentile.
False positive:  a high-income borrower who legitimately prepays in lumps and
                 looks "off" versus peers but is not risky.
False negative:  fraud that mimics normal repayment until a late sudden default.
Human-in-loop:   analyst reviews the reason list and confirms or dismisses;
                 decisions are stored and can feed back into tuning.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config

# Behavioural metrics used for peer comparison.
PEER_METRICS = [
    "repay_missed_ratio",
    "repay_mean_dpd",
    "repay_recent_missed",
    "bureau_mean_utilisation",
    "bureau_writeoff_ratio",
    "repay_total_shortfall",
]
PEER_KEYS = ["loan_type", "income_band"]


def _peer_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each metric within its peer segment."""
    out = df.copy()
    for m in PEER_METRICS:
        grp = out.groupby(PEER_KEYS, observed=True)[m]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        out[f"z_{m}"] = ((out[m] - mean) / std).fillna(0)
    return out


def _reasons(row: pd.Series) -> List[str]:
    """Human-readable explanations for why a row was flagged."""
    reasons = []
    labels = {
        "z_repay_missed_ratio": "missed-payment ratio",
        "z_repay_mean_dpd": "average days-past-due",
        "z_repay_recent_missed": "recent missed payments",
        "z_bureau_mean_utilisation": "bureau credit utilisation",
        "z_bureau_writeoff_ratio": "bureau write-off ratio",
        "z_repay_total_shortfall": "total payment shortfall",
    }
    for zcol, label in labels.items():
        z = row.get(zcol, 0)
        if z >= 2.5:
            reasons.append(f"{label} is {z:.1f} sd above its peer segment")
    if row.get("iforest_outlier", 0) == 1:
        reasons.append("multivariate behaviour pattern is an outlier (IsolationForest)")
    if not reasons:
        reasons.append("composite behaviour score above the 95th percentile")
    return reasons


def detect(df: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    if df is None:
        df = pd.read_parquet(config.SCORED_ACCOUNTS_FILE)

    z = _peer_zscores(df)
    zcols = [f"z_{m}" for m in PEER_METRICS]

    # Multivariate outlier detection on the raw behavioural metrics.
    iso = IsolationForest(
        n_estimators=200, contamination=0.05, random_state=config.RANDOM_STATE
    )
    feats = df[PEER_METRICS].fillna(0).values
    z["iforest_score"] = -iso.fit(feats).score_samples(feats)  # higher = more anomalous
    z["iforest_outlier"] = (iso.predict(feats) == -1).astype(int)

    # Composite score: max peer z (behaviour spikes) + scaled iforest score.
    z["peer_z_max"] = z[zcols].abs().max(axis=1)
    z["anomaly_score"] = z["peer_z_max"] + (
        (z["iforest_score"] - z["iforest_score"].mean()) / z["iforest_score"].std()
    )
    threshold = z["anomaly_score"].quantile(0.95)
    z["is_anomaly"] = (z["anomaly_score"] >= threshold).astype(int)

    flagged = z[z["is_anomaly"] == 1].copy()
    flagged["reasons"] = flagged.apply(_reasons, axis=1)
    flagged["reason_text"] = flagged["reasons"].apply(lambda r: "; ".join(r))
    flagged = flagged.sort_values("anomaly_score", ascending=False)

    cols = (
        ["account_id", "customer_id", "loan_type", "income_band", "anomaly_score",
         "peer_z_max", "iforest_score", "reason_text", config.TARGET, "risk_score"]
        + PEER_METRICS
    )
    result = flagged[[c for c in cols if c in flagged.columns]].reset_index(drop=True)

    if save:
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        result.to_parquet(config.ANOMALIES_FILE, index=False)
    return result


if __name__ == "__main__":
    res = detect(save=True)
    print(f"Flagged {len(res)} accounts.")
    print(res[["account_id", "anomaly_score", "reason_text"]].head(10).to_string())
