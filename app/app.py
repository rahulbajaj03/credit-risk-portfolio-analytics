"""Credit Risk Portfolio Analytics - Streamlit dashboard.

Views:
    * Portfolio      - KPIs, risk segments, exposure and default trends across
                       income / employment / region / age / loan type, with a
                       working filter.
    * Account        - per-account risk score and key SHAP drivers, drill-down.
    * Ask Portfolio  - LLM Q&A grounded in the real aggregates (AI feature 1).
    * Anomaly Review - review list of flagged accounts with reasons and a
                       confirm / dismiss human-in-the-loop step (AI feature 2).

Run:  streamlit run app/app.py
The app auto-builds artefacts on first run if they are missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Make 'src' importable whether run from repo root or elsewhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import analytics, config, llm_assistant  # noqa: E402

st.set_page_config(page_title="Credit Risk Portfolio", layout="wide", page_icon="📊")


# ---------------------------------------------------------------------------
# Data loading (cached). Auto-build if artefacts are missing.
# ---------------------------------------------------------------------------
def _ensure_artefacts() -> None:
    if not config.SCORED_ACCOUNTS_FILE.exists() or not config.ANOMALIES_FILE.exists():
        with st.spinner("First run: building datasets, model and anomalies ..."):
            from src import build_all

            build_all.main()


@st.cache_data(show_spinner=False)
def load_scored() -> pd.DataFrame:
    return pd.read_parquet(config.SCORED_ACCOUNTS_FILE)


@st.cache_data(show_spinner=False)
def load_anomalies() -> pd.DataFrame:
    return pd.read_parquet(config.ANOMALIES_FILE)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    import json

    return json.load(open(config.METRICS_FILE))


@st.cache_data(show_spinner=False)
def load_shap() -> dict:
    import json

    return json.load(open(config.SHAP_VALUES_FILE))


@st.cache_data(show_spinner=False)
def load_facts() -> dict:
    return llm_assistant.build_facts(load_scored())


_ensure_artefacts()
df = load_scored()
metrics = load_metrics()

PCT = lambda x: f"{x*100:.1f}%"  # noqa: E731


# ---------------------------------------------------------------------------
# Sidebar: navigation + global filter
# ---------------------------------------------------------------------------
st.sidebar.title("Credit Risk Portfolio")
view = st.sidebar.radio(
    "View",
    ["Portfolio", "Account detail", "Ask the Portfolio (AI)", "Anomaly review (AI)", "Model & monitoring"],
)

st.sidebar.markdown("### Filters")
loan_types = sorted(df["loan_type"].dropna().unique())
regions = sorted(df["region"].dropna().unique())
sel_loan = st.sidebar.multiselect("Loan type", loan_types, default=loan_types)
sel_region = st.sidebar.multiselect("Region", regions, default=regions)

fdf = df[df["loan_type"].isin(sel_loan) & df["region"].isin(sel_region)]
if fdf.empty:
    st.warning("No accounts match the current filter.")
    st.stop()

llm_on = llm_assistant._llm_available()
st.sidebar.markdown("---")
st.sidebar.caption(
    f"LLM: {'🟢 live' if llm_on else '🟡 fallback (no API key)'}"
)


# ===========================================================================
# PORTFOLIO VIEW
# ===========================================================================
if view == "Portfolio":
    st.title("Portfolio overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts", f"{len(fdf):,}")
    c2.metric("Default rate", PCT(fdf[config.TARGET].mean()))
    c3.metric("Total exposure", f"${fdf['loan_amount'].sum():,.0f}")
    c4.metric(
        "Defaulted exposure",
        f"${fdf.loc[fdf[config.TARGET]==1,'loan_amount'].sum():,.0f}",
    )

    st.markdown("#### Risk segments (model risk bands)")
    band = (
        fdf.groupby("risk_band", observed=True)
        .agg(accounts=("account_id", "count"), exposure=("loan_amount", "sum"),
             actual_default=(config.TARGET, "mean"))
        .reset_index()
    )
    b1, b2 = st.columns(2)
    b1.plotly_chart(
        px.bar(band, x="risk_band", y="accounts", color="risk_band",
               title="Accounts by risk band"),
        use_container_width=True,
    )
    b2.plotly_chart(
        px.bar(band, x="risk_band", y="actual_default", color="risk_band",
               title="Actual default rate by risk band (model calibration)"),
        use_container_width=True,
    )

    st.markdown("#### Default rate across dimensions")
    dim = st.selectbox(
        "Dimension",
        ["loan_type", "employment_type", "region", "income_band", "age_band"],
    )
    cut = analytics.default_rate_by(fdf, dim)
    g1, g2 = st.columns([2, 1])
    g1.plotly_chart(
        px.bar(cut, x=dim, y="default_rate", color="default_rate",
               color_continuous_scale="Reds", title=f"Default rate by {dim}"),
        use_container_width=True,
    )
    g2.dataframe(
        cut.assign(default_rate=lambda d: (d["default_rate"] * 100).round(1)),
        use_container_width=True, hide_index=True,
    )

    st.markdown("#### Cross-dimensional cut: employment × income band")
    ct = analytics.crosstab_default(fdf, "employment_type", "income_band")
    st.plotly_chart(
        px.imshow((ct * 100).round(1), text_auto=True, color_continuous_scale="Reds",
                  aspect="auto", labels=dict(color="default %")),
        use_container_width=True,
    )


# ===========================================================================
# ACCOUNT DETAIL VIEW
# ===========================================================================
elif view == "Account detail":
    st.title("Account detail")

    sort_desc = st.checkbox("Show highest-risk accounts first", value=True)
    view_df = fdf.sort_values("risk_score", ascending=not sort_desc)
    acct_id = st.selectbox("Select account", view_df["account_id"].tolist())
    row = fdf[fdf["account_id"] == acct_id]

    r = row.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk score", PCT(r["risk_score"]))
    c2.metric("Risk band", str(r["risk_band"]))
    c3.metric("Loan type", r["loan_type"])
    c4.metric("Actual status", "Defaulted" if r[config.TARGET] == 1 else "Performing")

    st.markdown("#### Key drivers (SHAP)")
    try:
        from src import model as model_mod

        drivers = model_mod.explain_account(row)
        ddf = pd.DataFrame(drivers)
        st.plotly_chart(
            px.bar(ddf, x="shap", y="feature", orientation="h", color="shap",
                   color_continuous_scale="RdBu_r",
                   title="Why this account scored the way it did"),
            use_container_width=True,
        )
        st.dataframe(ddf, use_container_width=True, hide_index=True)
    except Exception as e:  # noqa: BLE001
        st.info(f"SHAP explanation unavailable: {e}")

    st.markdown("#### Account attributes")
    show_cols = [
        "customer_id", "loan_amount", "interest_rate_pct", "tenure_months",
        "annual_income", "employment_type", "region", "bureau_n_written_off",
        "repay_n_missed", "repay_mean_dpd", "coll_n_events",
    ]
    st.dataframe(row[show_cols].T.rename(columns={row.index[0]: "value"}),
                 use_container_width=True)


# ===========================================================================
# ASK THE PORTFOLIO (LLM)
# ===========================================================================
elif view == "Ask the Portfolio (AI)":
    st.title("Ask the Portfolio")
    st.caption(
        "Natural-language Q&A grounded in the real portfolio aggregates. "
        "Answers quote exact figures and their source; questions the data "
        "can't answer are declined."
    )
    if not llm_on:
        st.info(
            "No LLM API key is set, so answers come from a deterministic "
            "fallback that quotes the same real figures. Set OPENAI_API_KEY "
            "(in Streamlit secrets) to enable the full LLM.",
            icon="🟡",
        )

    with st.expander("What triggers this feature / false-positive & -negative / human step"):
        st.markdown(
            "- **Trigger:** analyst submits a question.\n"
            "- **False positive:** the model answers something the data can't "
            "support — mitigated by the strict 'answer only from facts' prompt "
            "and the decline rule.\n"
            "- **False negative:** a valid but oddly-phrased question isn't "
            "matched — the analyst can rephrase; the facts pack is shown below.\n"
            "- **Human-in-the-loop:** the analyst reads the cited figure and "
            "decides whether to trust/act on it."
        )

    examples = [
        "What is the portfolio default rate?",
        "Which loan type has the highest default rate?",
        "Which employment type is riskiest?",
        "What is the total exposure?",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, use_container_width=True):
            st.session_state["q"] = ex

    q = st.text_input("Your question", value=st.session_state.get("q", ""))
    if st.button("Ask", type="primary") or st.session_state.get("q"):
        if q.strip():
            facts = load_facts()
            out = llm_assistant.ask(q, facts)
            st.markdown("#### Answer")
            st.success(out["answer"])
            st.caption(f"engine: {out['engine']}")
        st.session_state["q"] = ""

    with st.expander("Grounding facts the assistant may use (full transparency)"):
        st.json(load_facts())


# ===========================================================================
# ANOMALY REVIEW
# ===========================================================================
elif view == "Anomaly review (AI)":
    st.title("Anomaly review")
    anom = load_anomalies()
    anom = anom[anom["loan_type"].isin(sel_loan)]

    st.caption(
        "Accounts whose repayment / bureau behaviour is unusual versus their "
        "peer segment (loan type × income band) and/or flagged by an "
        "IsolationForest. Review the reason, then confirm or dismiss."
    )
    with st.expander("What triggers this feature / false-positive & -negative / human step"):
        st.markdown(
            "- **Trigger:** composite anomaly score (peer z-scores + "
            "IsolationForest) above the 95th percentile.\n"
            "- **False positive:** a wealthy borrower who prepays in lumps "
            "looks 'off' versus peers but isn't risky.\n"
            "- **False negative:** behaviour that mimics normal repayment until "
            "a sudden late default.\n"
            "- **Human-in-the-loop:** the analyst confirms or dismisses each "
            "flag below; decisions are recorded for the session and could feed "
            "back into tuning."
        )

    if "decisions" not in st.session_state:
        st.session_state["decisions"] = {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged accounts", f"{len(anom):,}")
    c2.metric("Reviewed", f"{len(st.session_state['decisions']):,}")
    confirmed = sum(1 for v in st.session_state["decisions"].values() if v == "Confirmed")
    c3.metric("Confirmed", f"{confirmed:,}")

    top_n = st.slider("Show top N by anomaly score", 5, 50, 15)
    for _, a in anom.head(top_n).iterrows():
        aid = a["account_id"]
        with st.container(border=True):
            left, right = st.columns([3, 1])
            left.markdown(
                f"**{aid}** · {a['loan_type']} · income {a['income_band']} · "
                f"score **{a['anomaly_score']:.1f}** · "
                f"actual: {'Defaulted' if a[config.TARGET]==1 else 'Performing'}"
            )
            left.caption(a["reason_text"])
            decision = st.session_state["decisions"].get(aid, "Pending")
            right.write(f"Decision: **{decision}**")
            b1, b2 = right.columns(2)
            if b1.button("Confirm", key=f"c_{aid}"):
                st.session_state["decisions"][aid] = "Confirmed"
                st.rerun()
            if b2.button("Dismiss", key=f"d_{aid}"):
                st.session_state["decisions"][aid] = "Dismissed"
                st.rerun()


# ===========================================================================
# MODEL & MONITORING
# ===========================================================================
elif view == "Model & monitoring":
    st.title("Model performance & monitoring")

    lr, xgb = metrics["logistic_regression"], metrics["xgboost"]
    st.markdown("#### Test-set metrics")
    mt = pd.DataFrame(
        {
            "Model": ["Logistic Regression (baseline)", "XGBoost (challenger)"],
            "AUC": [lr["auc"], xgb["auc"]],
            "Gini": [lr["gini"], xgb["gini"]],
            "KS": [lr["ks"], xgb["ks"]],
        }
    ).round(4)
    st.dataframe(mt, use_container_width=True, hide_index=True)

    st.markdown("#### Global feature importance (XGBoost, mean |SHAP|)")
    shap = load_shap()
    sdf = pd.DataFrame({"feature": shap["feature"], "mean_abs_shap": shap["mean_abs_shap"]})
    st.plotly_chart(
        px.bar(sdf.head(15).iloc[::-1], x="mean_abs_shap", y="feature",
               orientation="h", title="Top drivers of predicted default"),
        use_container_width=True,
    )

    st.markdown("#### Post-deployment monitoring plan")
    st.markdown(
        """
- **Discrimination drift:** track AUC / KS / Gini monthly on newly-matured
  accounts; alert if KS drops >10% from the training baseline.
- **Population Stability Index (PSI):** on the score distribution and on the
  top features; PSI > 0.25 signals a material population shift → investigate.
- **Calibration:** compare predicted vs realised default rate per risk band;
  recalibrate if bands drift.
- **Data-quality gates:** monitor null rates on the engineered features
  (interest rate, DPD) against the levels seen in training.
- **Fairness / stability:** monitor approval and default rates across region
  and employment segments for unintended disparate impact.
- **Refresh cadence:** scheduled quarterly retrain + ad-hoc retrain when a PSI
  or KS alert fires, with champion/challenger comparison before promotion.
        """
    )
