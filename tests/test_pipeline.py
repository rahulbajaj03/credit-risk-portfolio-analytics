"""Smoke / sanity tests for the pipeline.

Run with:  python -m pytest -q
These assume `python -m src.build_all` has produced the artefacts.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import analytics, config, llm_assistant


@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    if not config.SCORED_ACCOUNTS_FILE.exists():
        pytest.skip("Artefacts not built. Run `python -m src.build_all` first.")
    return pd.read_parquet(config.SCORED_ACCOUNTS_FILE)


def test_account_grain_unique(scored):
    assert scored["account_id"].is_unique


def test_default_rate_plausible(scored):
    rate = scored[config.TARGET].mean()
    assert 0.10 < rate < 0.25  # brief says ~16%


def test_risk_scores_in_range(scored):
    assert scored["risk_score"].between(0, 1).all()


def test_no_target_leakage_in_features():
    from src import model

    assert config.TARGET not in model.NUMERIC_FEATURES
    assert config.TARGET not in model.CATEGORICAL_FEATURES


def test_metrics_reasonable():
    import json

    if not config.METRICS_FILE.exists():
        pytest.skip("Metrics not built.")
    m = json.load(open(config.METRICS_FILE))
    assert m["xgboost"]["auc"] > 0.6
    assert m["logistic_regression"]["auc"] > 0.6


def test_llm_grounding_declines_offtopic(scored):
    facts = llm_assistant.build_facts(scored)
    out = llm_assistant.ask("What is the capital of France?", facts)
    assert "can't answer" in out["answer"].lower()


def test_llm_answers_from_facts(scored):
    facts = llm_assistant.build_facts(scored)
    out = llm_assistant.ask("Which loan type has the highest default rate?", facts)
    assert "%" in out["answer"]


def test_anomalies_have_reasons():
    if not config.ANOMALIES_FILE.exists():
        pytest.skip("Anomalies not built.")
    anom = pd.read_parquet(config.ANOMALIES_FILE)
    assert len(anom) > 0
    assert anom["reason_text"].str.len().gt(0).all()


def test_facts_match_source(scored):
    """LLM facts must equal the analytics cut (no drift / hallucination risk)."""
    facts = llm_assistant.build_facts(scored)
    cut = analytics.default_rate_by(scored, "loan_type")
    assert facts["default_rate_by_loan_type"][0]["loan_type"] == cut.iloc[0]["loan_type"]
