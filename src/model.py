"""Default-prediction model.

Trains two models on engineered bureau + repayment + collections features:

    * Logistic regression  - transparent baseline
    * XGBoost              - gradient-boosted challenger (the headline model)

Reports AUC, KS and Gini for both, computes SHAP values for the XGBoost model
for global + per-account explanations, and scores every account so the
dashboard can show a risk score and its key drivers.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config

# ---------------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: List[str] = [
    # loan
    "loan_amount", "interest_rate_pct", "tenure_months", "collateral_flag",
    "loan_age_months", "interest_rate_was_missing",
    # customer
    "age", "annual_income", "years_employed", "num_dependents",
    "years_at_residence",
    # bureau
    "bureau_n_lines", "bureau_n_written_off", "bureau_n_active",
    "bureau_total_sanctioned", "bureau_total_outstanding", "bureau_max_dpd",
    "bureau_mean_dpd", "bureau_mean_utilisation", "bureau_writeoff_ratio",
    "bureau_overall_utilisation",
    # repayment
    "repay_n_months", "repay_n_missed", "repay_n_late", "repay_max_dpd",
    "repay_mean_dpd", "repay_total_shortfall", "repay_missed_ratio",
    "repay_late_ratio", "repay_recent_missed", "repay_recent_dpd",
    # collections
    "coll_n_events", "coll_total_recovered", "coll_n_legal",
    "coll_n_no_response", "has_collections",
]
CATEGORICAL_FEATURES: List[str] = [
    "loan_type", "employment_type", "region", "city_tier", "housing_type",
    "education_level", "gender", "marital_status",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max gap between cumulative good/bad distributions."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def score_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y_true, y_score)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, y_score),
    }


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[config.TARGET].astype(int)
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )


def _preprocessor() -> ColumnTransformer:
    num = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", num, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)]
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(save: bool = True) -> Dict:
    df = pd.read_parquet(config.ACCOUNT_DATASET)
    X_train, X_test, y_train, y_test = _split(df)

    # --- Logistic regression baseline -------------------------------------
    logreg = Pipeline(
        [
            ("pre", _preprocessor()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    logreg.fit(X_train, y_train)
    logreg_scores = logreg.predict_proba(X_test)[:, 1]
    logreg_metrics = score_metrics(y_test.values, logreg_scores)

    # --- XGBoost challenger -----------------------------------------------
    # XGBoost handles numerics natively; we one-hot the categoricals through
    # the same preprocessor so SHAP maps cleanly to readable feature names.
    pre = _preprocessor()
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    feature_names = _feature_names(pre)

    # Note on tuning: the signal in this portfolio is genuinely subtle (the
    # strongest single-feature correlation with default is ~0.16), so shallow,
    # heavily-regularised trees generalise best. We deliberately do NOT use
    # scale_pos_weight here: up-weighting the positive class distorts the
    # predicted probabilities and hurt ranking/AUC in testing. Class imbalance
    # is instead addressed downstream via the risk-band thresholds.
    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.015,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        min_child_weight=10,
        eval_metric="auc",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train_t, y_train)
    xgb_scores = xgb.predict_proba(X_test_t)[:, 1]
    xgb_metrics = score_metrics(y_test.values, xgb_scores)

    metrics = {
        "logistic_regression": logreg_metrics,
        "xgboost": xgb_metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "default_rate": float(df[config.TARGET].mean()),
    }

    # --- SHAP global importance -------------------------------------------
    shap_summary = _shap_global(xgb, X_test_t, feature_names)

    # --- Score every account for the dashboard ----------------------------
    X_all_t = pre.transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    df_scored = df.copy()
    df_scored["risk_score"] = xgb.predict_proba(X_all_t)[:, 1]
    df_scored["risk_band"] = pd.cut(
        df_scored["risk_score"],
        bins=[-0.001, 0.1, 0.25, 0.5, 1.0],
        labels=["Low", "Medium", "High", "Very High"],
    )

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": logreg}, config.LOGREG_FILE)
        joblib.dump(
            {"preprocessor": pre, "model": xgb, "feature_names": feature_names},
            config.MODEL_FILE,
        )
        with open(config.FEATURE_LIST_FILE, "w") as f:
            json.dump(
                {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
                f,
                indent=2,
            )
        with open(config.METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
        with open(config.SHAP_VALUES_FILE, "w") as f:
            json.dump(shap_summary, f, indent=2)
        df_scored.to_parquet(config.SCORED_ACCOUNTS_FILE, index=False)

    return {"metrics": metrics, "shap": shap_summary, "scored": df_scored}


def _feature_names(pre: ColumnTransformer) -> List[str]:
    names = list(NUMERIC_FEATURES)
    oh = pre.named_transformers_["cat"].named_steps["oh"]
    names += list(oh.get_feature_names_out(CATEGORICAL_FEATURES))
    return names


def _shap_global(model, X_t, feature_names: List[str]) -> Dict:
    """Mean |SHAP| per feature (global importance), top 20."""
    import shap

    explainer = shap.TreeExplainer(model)
    # Sample for speed if large.
    sample = X_t if X_t.shape[0] <= 3000 else X_t[:3000]
    sv = explainer.shap_values(sample)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:20]
    return {
        "feature": [feature_names[i] for i in order],
        "mean_abs_shap": [float(mean_abs[i]) for i in order],
    }


# ---------------------------------------------------------------------------
# Per-account explanation (used by the dashboard account view)
# ---------------------------------------------------------------------------
def explain_account(account_row: pd.DataFrame, top_n: int = 6) -> List[dict]:
    """Return the top SHAP drivers for a single account."""
    import shap

    bundle = joblib.load(config.MODEL_FILE)
    pre, model, feature_names = bundle["preprocessor"], bundle["model"], bundle["feature_names"]
    X_t = pre.transform(account_row[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_t)[0]
    order = np.argsort(np.abs(sv))[::-1][:top_n]
    return [
        {
            "feature": feature_names[i],
            "shap": float(sv[i]),
            "direction": "increases risk" if sv[i] > 0 else "lowers risk",
        }
        for i in order
    ]


if __name__ == "__main__":
    out = train(save=True)
    m = out["metrics"]
    print("Logistic regression:", m["logistic_regression"])
    print("XGBoost:            ", m["xgboost"])
    print("Top SHAP features:  ", out["shap"]["feature"][:8])
