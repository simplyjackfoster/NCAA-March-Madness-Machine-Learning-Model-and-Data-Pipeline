from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.common.config import load_config
from src.common.io import write_json


FEATURES = ["elo_diff", "net_rating_diff", "tempo_diff"]


def run_loyo(config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    train = pd.read_parquet(root / "data" / "processed" / "train.parquet").copy()

    if "season" not in train.columns:
        raise ValueError("LOYO requires a 'season' column in training data.")

    metrics = []
    for season in sorted(train["season"].unique()):
        tr = train[train["season"] != season]
        va = train[train["season"] == season]
        if tr.empty or va.empty:
            continue

        model = LogisticRegression(max_iter=1000)
        model.fit(tr[FEATURES], tr["label"])
        probs = model.predict_proba(va[FEATURES])[:, 1]
        metrics.append(
            {
                "holdout_season": int(season),
                "log_loss": float(log_loss(va["label"], probs)),
                "brier": float(brier_score_loss(va["label"], probs)),
                "n_validation": int(len(va)),
            }
        )

    if not metrics:
        # Single-season fallback for bootstrap runs.
        cut = int(len(train) * 0.8)
        tr, va = train.iloc[:cut], train.iloc[cut:]
        model = LogisticRegression(max_iter=1000)
        model.fit(tr[FEATURES], tr["label"])
        probs = model.predict_proba(va[FEATURES])[:, 1]
        metrics.append(
            {
                "holdout_season": "pseudo_split",
                "log_loss": float(log_loss(va["label"], probs)),
                "brier": float(brier_score_loss(va["label"], probs)),
                "n_validation": int(len(va)),
            }
        )

    summary = {
        "folds": metrics,
        "mean_log_loss": float(sum(m["log_loss"] for m in metrics) / len(metrics)),
        "mean_brier": float(sum(m["brier"] for m in metrics) / len(metrics)),
        "n_folds": len(metrics),
    }
    out = root / "outputs" / "validation" / "loyo_results.json"
    write_json(out, summary)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {run_loyo(args.config)}")
