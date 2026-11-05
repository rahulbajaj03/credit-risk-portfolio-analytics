# Credit Risk Portfolio Analytics

An interactive credit-risk monitoring dashboard built from 5 synthetic bank
tables, with two embedded AI features: an LLM **"Ask the Portfolio"** assistant
and **anomaly detection**.

**🔗 Live app:** https://credit-risk-portfolio-analytics-7mhhduvp4f4exfbohmwmou.streamlit.app/
(opens with no login; the LLM feature is live)

**Python version:** 3.10 (also works on 3.11; the hosted app targets 3.11 on
Streamlit Community Cloud).

## Run it in three commands

```bash
pip install -r requirements.txt          # 1. install
python -m src.build_all                   # 2. build the clean data + model + anomalies
streamlit run app/app.py                  # 3. start the dashboard
```

The dashboard also auto-builds the artefacts on first launch if they are
missing, so step 2 is optional locally but recommended (and required for the
tests).

## LLM API key

The "Ask the Portfolio" feature uses an OpenAI-compatible LLM.

- **Locally:** copy `.env.example` to `.env` and set `OPENAI_API_KEY`, or
  export it in your shell.
- **On Streamlit Community Cloud / HF Spaces:** add `OPENAI_API_KEY` in the
  app's **Secrets** store (see `.streamlit/secrets.toml.example`). Never commit
  the real key.
- Optional overrides: `OPENAI_BASE_URL` (any OpenAI-compatible endpoint) and
  `LLM_MODEL` (default `gpt-4o-mini`).

**If no key is set,** the app still runs. "Ask the Portfolio" switches to a
deterministic fallback that answers the common questions directly from the same
real aggregated figures (so numbers are never hallucinated), and the sidebar
shows a 🟡 fallback indicator.

## Where each item in the brief lives

| Brief item | Location |
|---|---|
| 1. Data ingestion + join + DQ decisions | `src/ingest.py`; DQ log → `data/processed/data_quality_report.json` |
| 2. Portfolio analytics / profiling | `src/analytics.py`; Portfolio view in `app/app.py` |
| 3. Default model (LogReg + XGBoost, AUC/KS/Gini, SHAP, monitoring) | `src/model.py`; "Model & monitoring" view |
| 4. Dashboard (portfolio + account views, filter, both AI features) | `app/app.py` |
| AI feature 1 — Ask the Portfolio (LLM) | `src/llm_assistant.py`; "Ask the Portfolio" view |
| AI feature 2 — Anomaly detection | `src/anomaly.py`; "Anomaly review" view |
| The 5 raw CSVs | `data/raw/` |
| Architecture diagram | `docs/architecture.pdf` |
| Solution overview | `docs/solution_overview.pdf` |
| Tests | `tests/` (`python -m pytest -q`) |

## Repository layout

```
README.md
requirements.txt          # every package pinned
.env.example              # names of required keys, no values
app/app.py                # dashboard entry point
src/                      # ingestion, analytics, model, AI features
  config.py               # paths + constants
  ingest.py               # load, clean, join -> account_master / customer_master
  analytics.py            # profiling cuts + KPIs
  model.py                # LogReg baseline + XGBoost + SHAP + scoring
  anomaly.py              # peer-z + IsolationForest anomaly detection
  llm_assistant.py        # grounded LLM Q&A + no-key fallback
  build_all.py            # one-shot pipeline
data/raw/                 # the 5 CSVs as provided
data/processed/           # generated artefacts (gitignored)
models/                   # generated model artefacts (gitignored)
docs/architecture.pdf
docs/solution_overview.pdf
tests/
```

## Model results (test set)

| Model | AUC | Gini | KS |
|---|---|---|---|
| Logistic Regression (baseline) | ~0.67 | ~0.34 | ~0.25 |
| XGBoost (challenger) | ~0.67 | ~0.33 | ~0.25 |

The signal in this synthetic portfolio is genuinely subtle (the strongest
single-feature correlation with default is ~0.16), so AUC in the high-0.60s is
the honest ceiling. XGBoost edges the baseline on KS while staying
well-calibrated. Top drivers (SHAP): annual income, years employed, repayment
days-past-due, age, and bureau delinquency — consistent with the brief.

## Notes on data-quality handling

All handled explicitly (see `src/ingest.py` and the generated DQ report), never
silently dropped:

- **Inconsistent gender labels** (`M`/`F` → `Male`/`Female`).
- **Missing `years_employed`, `num_dependents`, `years_at_residence`:**
  median-imputed with a `*_was_missing` flag kept.
- **Missing `interest_rate_pct`:** imputed by per-`loan_type` median (secured
  vs unsecured price differently) with a missing flag.
- **Missing bureau `days_past_due`:** filled with 0 (no delinquency reported).
- **Sparse collections:** accounts with no events get 0s plus a
  `has_collections` flag.
