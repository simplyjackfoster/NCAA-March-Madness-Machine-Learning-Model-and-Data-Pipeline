from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.common.config import load_config
from src.models.diagnostics import write_calibration_diagnostics


def train_calibrator(config_path: str = "configs/config.yaml"):
    """Fit Platt scaling (logistic regression) on cross-val probability estimates.

    Maps raw model output probabilities to calibrated probabilities so that
    a 70% prediction actually wins ~70% of the time.
    """
    cfg = load_config(config_path)
    root = cfg["_root"]
    cal = pd.read_parquet(root / "data" / "processed" / "calibration_set.parquet")

    probs = cal["model_prob"].to_numpy().reshape(-1, 1)
    outcomes = cal["outcome"].to_numpy()

    # Platt scaling: logistic regression mapping [0,1] -> [0,1]
    platt = LogisticRegression(max_iter=1000)
    platt.fit(probs, outcomes)

    # File is named isotonic.pkl for backwards-compatibility with downstream consumers.
    # The object stored is now a Platt (logistic regression) scaler, not an isotonic regressor.
    out = root / "artifacts" / "calibrators" / "isotonic.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(platt, f)

    calibrated = platt.predict_proba(probs)[:, 1]
    write_calibration_diagnostics(
        outcomes, calibrated, root / "outputs" / "validation"
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {train_calibrator(args.config)}")
