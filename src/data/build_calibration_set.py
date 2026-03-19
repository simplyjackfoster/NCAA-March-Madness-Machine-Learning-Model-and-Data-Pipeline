from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from src.common.config import load_config

FEATURES = ["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]


def build_calibration_set(config_path: str = "configs/config.yaml") -> Path:
    """Generate out-of-fold probability estimates for Platt scaling calibration.

    Uses 5-fold cross-validation on the training data so that the predicted
    probabilities are unbiased (not overfit to the training set).
    """
    cfg = load_config(config_path)
    root = cfg["_root"]
    train = pd.read_parquet(root / "data" / "processed" / "train.parquet")

    x = train[FEATURES]
    y = train["label"]

    model = LogisticRegression(max_iter=1000, C=0.5, random_state=cfg["data"]["random_seed"])
    probs = cross_val_predict(model, x, y, cv=5, method="predict_proba")[:, 1]

    cal_df = pd.DataFrame({"model_prob": probs, "outcome": y.values})
    out = root / "data" / "processed" / "calibration_set.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cal_df.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {build_calibration_set(args.config)}")
