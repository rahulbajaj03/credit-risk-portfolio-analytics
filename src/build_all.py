"""One-shot build: ingest -> model -> anomalies. Produces all artefacts the
dashboard needs. Run with:  python -m src.build_all
"""
from __future__ import annotations

from . import anomaly, ingest, model


def main() -> None:
    print("[1/3] Building clean datasets ...")
    out = ingest.build_datasets(save=True)
    print(f"      account_master: {out['account_master'].shape}")

    print("[2/3] Training models (logistic regression + XGBoost) ...")
    m = model.train(save=True)
    print(f"      LogReg  -> {m['metrics']['logistic_regression']}")
    print(f"      XGBoost -> {m['metrics']['xgboost']}")

    print("[3/3] Detecting anomalies ...")
    res = anomaly.detect(save=True)
    print(f"      flagged {len(res)} accounts")
    print("Done. Artefacts are in data/processed/ and models/.")


if __name__ == "__main__":
    main()
