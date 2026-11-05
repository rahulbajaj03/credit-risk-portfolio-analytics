"""'Ask the Portfolio' assistant (AI feature 1).

Natural-language Q&A grounded in the *actual* aggregated portfolio data. To
guarantee no hallucinated numbers, we never ask the LLM to do arithmetic on
raw rows. Instead we:

    1. Pre-compute a compact 'facts pack' (KPIs + every profiling cut) from
       account_master using src.analytics -- these are real, exact figures.
    2. Pass that facts pack to the LLM as grounding context and instruct it to
       answer ONLY from those facts, cite the figure and its source table, and
       decline anything the facts cannot answer.

Trigger:        analyst submits a question in the dashboard.
False positive: LLM answers a question the data cannot support -> mitigated by
                the 'decline if not in facts' instruction + a deterministic
                fallback that only quotes retrieved facts.
False negative: a valid question phrased oddly fails to match -> analyst can
                rephrase; the facts pack is always shown for transparency.
Human-in-loop:  the analyst reads the answer with its cited source and decides
                whether to trust / act on it.

If no API key is configured, a deterministic keyword-matching fallback answers
the common questions directly from the facts pack (so the app still works).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import pandas as pd

from . import analytics, config


# ---------------------------------------------------------------------------
# Facts pack: real numbers the LLM is allowed to use.
# ---------------------------------------------------------------------------
def build_facts(df: Optional[pd.DataFrame] = None) -> Dict:
    if df is None:
        df = analytics.load_account_master()
    kpis = analytics.headline_kpis(df)
    cuts = analytics.portfolio_summary(df)

    def tbl_to_records(t: pd.DataFrame) -> list:
        return json.loads(t.to_json(orient="records"))

    facts = {
        "kpis": kpis,
        "default_rate_by_loan_type": tbl_to_records(cuts["by_loan_type"]),
        "default_rate_by_employment": tbl_to_records(cuts["by_employment"]),
        "default_rate_by_region": tbl_to_records(cuts["by_region"]),
        "default_rate_by_income_band": tbl_to_records(cuts["by_income_band"]),
        "default_rate_by_age_band": tbl_to_records(cuts["by_age_band"]),
        "employment_x_income_default_rate": json.loads(
            cuts["employment_x_income"].to_json()
        ),
    }
    return facts


def facts_as_text(facts: Dict) -> str:
    return json.dumps(facts, indent=2, default=str)


SYSTEM_PROMPT = """You are a credit-risk portfolio analyst assistant.
You answer ONLY using the JSON facts provided in the user message under
"PORTFOLIO FACTS". These facts are exact figures computed from the real data.

Rules:
- Never invent or estimate numbers. Only quote figures present in the facts.
- Always state the exact figure and which fact table it came from
  (e.g. "default_rate_by_loan_type").
- Express default rates as percentages to one decimal place.
- If the facts do not contain the answer, reply exactly:
  "I can't answer that from the available portfolio data." and briefly say what
  data would be needed.
- Keep answers short and plain-English (2-4 sentences).
"""


# ---------------------------------------------------------------------------
# LLM call (OpenAI-compatible). Configurable via env vars or Streamlit secrets.
# ---------------------------------------------------------------------------
def _get(name: str, default: str | None = None) -> Optional[str]:
    """Read config from env first, then Streamlit secrets if available.

    This lets the same code work locally (.env / shell) and on Streamlit
    Community Cloud / HF Spaces (secrets store) without changes.
    """
    val = os.getenv(name)
    if val:
        return val
    try:  # st.secrets raises if there is no secrets file, so guard it.
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # noqa: BLE001
        pass
    return default


def _llm_available() -> bool:
    return bool(_get("OPENAI_API_KEY"))


def _ask_llm(question: str, facts: Dict) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=_get("OPENAI_API_KEY"),
        base_url=_get("OPENAI_BASE_URL") or None,
    )
    model = _get("LLM_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"PORTFOLIO FACTS:\n{facts_as_text(facts)}\n\nQUESTION: {question}",
            },
        ],
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Deterministic fallback (no API key) - answers common questions from facts.
# ---------------------------------------------------------------------------
def _fallback(question: str, facts: Dict) -> str:
    q = question.lower()
    kpis = facts["kpis"]

    def top(records, label_key):
        r = records[0]
        return r[label_key], r["default_rate"]

    if "portfolio" in q and ("default" in q or "rate" in q):
        return (
            f"The overall portfolio default rate is "
            f"{kpis['portfolio_default_rate']*100:.1f}% across "
            f"{kpis['n_accounts']:,} accounts (source: kpis)."
        )
    if "loan type" in q or "loan_type" in q or "product" in q:
        name, rate = top(facts["default_rate_by_loan_type"], "loan_type")
        return (
            f"{name} has the highest default rate at {rate*100:.1f}% "
            f"(source: default_rate_by_loan_type)."
        )
    if "employment" in q or "job" in q:
        name, rate = top(facts["default_rate_by_employment"], "employment_type")
        return (
            f"{name} borrowers have the highest default rate at {rate*100:.1f}% "
            f"(source: default_rate_by_employment)."
        )
    if "region" in q:
        name, rate = top(facts["default_rate_by_region"], "region")
        return (
            f"The {name} region has the highest default rate at {rate*100:.1f}% "
            f"(source: default_rate_by_region)."
        )
    if "income" in q:
        name, rate = top(facts["default_rate_by_income_band"], "income_band")
        return (
            f"The {name} income band has the highest default rate at {rate*100:.1f}% "
            f"(source: default_rate_by_income_band)."
        )
    if "age" in q:
        name, rate = top(facts["default_rate_by_age_band"], "age_band")
        return (
            f"The {name} age band has the highest default rate at {rate*100:.1f}% "
            f"(source: default_rate_by_age_band)."
        )
    if "exposure" in q:
        return (
            f"Total exposure is ${kpis['total_exposure']:,.0f}, of which "
            f"${kpis['defaulted_exposure']:,.0f} sits in defaulted accounts "
            f"(source: kpis)."
        )
    return (
        "I can't answer that from the available portfolio data. "
        "Try asking about default rates by loan type, employment, region, "
        "income band, age band, or total exposure."
    )


def ask(question: str, facts: Optional[Dict] = None) -> Dict[str, str]:
    """Return an answer plus which engine produced it."""
    if facts is None:
        facts = build_facts()
    if not question or not question.strip():
        return {"answer": "Please enter a question.", "engine": "none"}
    if _llm_available():
        try:
            return {"answer": _ask_llm(question, facts), "engine": "llm"}
        except Exception as e:  # noqa: BLE001 - surface gracefully in the UI
            return {
                "answer": _fallback(question, facts)
                + f"\n\n(Note: LLM call failed: {e}. Showing deterministic answer.)",
                "engine": "fallback-after-error",
            }
    return {"answer": _fallback(question, facts), "engine": "fallback-no-key"}


if __name__ == "__main__":
    f = build_facts()
    for q in [
        "What is the portfolio default rate?",
        "Which loan type has the highest default rate?",
        "Which employment type is riskiest?",
        "What's the capital of France?",
    ]:
        print(f"Q: {q}\nA: {ask(q, f)['answer']}\n")
