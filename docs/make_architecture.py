"""Generate docs/architecture.pdf - a one-page data-flow diagram showing
raw data -> ingestion -> analytics -> model -> dashboard, with both AI
features and the tool used at each stage.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "architecture.pdf"


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2, edgecolor="#333", facecolor=color, alpha=0.9,
        )
    )
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top",
            fontsize=10, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.42, "\n".join(lines), ha="center", va="top",
            fontsize=7.4, color="#222")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.3, color="#444")
    )


def main():
    fig, ax = plt.subplots(figsize=(11.0, 8.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(6, 8.7, "Credit Risk Portfolio Analytics — Architecture",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(6, 8.35, "raw data → ingestion → analytics → model → dashboard (with both AI features)",
            ha="center", fontsize=9, color="#555")

    # Row 1: pipeline
    box(ax, 0.3, 6.2, 2.2, 1.6, "1. Raw data",
        ["5 CSVs (data/raw)", "customers, loans,", "bureau, repayment,", "collections",
         "", "tool: pandas"], "#e8eef7")
    box(ax, 2.9, 6.2, 2.4, 1.6, "2. Ingestion",
        ["src/ingest.py", "clean DQ issues,", "join on customer_id /", "account_id, engineer",
         "features", "tool: pandas → parquet"], "#e3f0e6")
    box(ax, 5.7, 6.2, 2.4, 1.6, "3. Analytics",
        ["src/analytics.py", "KPIs + profiling cuts", "(income, employment,", "region, age, loan type)",
         "", "tool: pandas"], "#fdf1dc")
    box(ax, 8.5, 6.2, 3.2, 1.6, "4. Model",
        ["src/model.py", "LogReg baseline +", "XGBoost + SHAP", "AUC / KS / Gini,",
         "scores every account", "tool: scikit-learn, xgboost, shap"], "#f6e3ef")

    arrow(ax, 2.5, 7.0, 2.9, 7.0)
    arrow(ax, 5.3, 7.0, 5.7, 7.0)
    arrow(ax, 8.1, 7.0, 8.5, 7.0)

    # down to dashboard
    arrow(ax, 6.0, 6.2, 6.0, 5.1)

    # Dashboard
    box(ax, 1.5, 3.2, 9.0, 1.8, "5. Dashboard  (app/app.py — Streamlit, hosted)",
        ["Portfolio view: risk segments, exposure, default trends across dimensions, filters",
         "Account view: risk score + key SHAP drivers, drill-down",
         "Embeds both AI features below  ·  reads scored_accounts.parquet + anomalies.parquet",
         "tool: Streamlit + Plotly"], "#eef4fb")

    # AI features
    box(ax, 1.5, 0.8, 4.2, 1.9, "AI feature 1 — Ask the Portfolio (LLM)",
        ["src/llm_assistant.py", "Pre-computed FACTS pack (exact aggregates)", "sent to LLM as grounding →",
         "answers quote figure + source;", "declines if not in facts.",
         "No-key fallback = deterministic.", "tool: OpenAI-compatible API"], "#fff3f3")
    box(ax, 6.3, 0.8, 4.2, 1.9, "AI feature 2 — Anomaly detection",
        ["src/anomaly.py", "peer z-scores (loan_type×income)", "+ IsolationForest →",
         "flagged review list with reasons;", "analyst confirms / dismisses.",
         "", "tool: scikit-learn"], "#f3fff5")

    arrow(ax, 3.6, 3.2, 3.6, 2.7)
    arrow(ax, 8.4, 3.2, 8.4, 2.7)

    # LLM data note
    ax.text(6, 0.35,
            "LLM data boundary: only the aggregated FACTS pack (KPIs + profiling cuts) is sent to the LLM — never raw customer rows or PII.",
            ha="center", fontsize=7.6, color="#a11", style="italic")

    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
