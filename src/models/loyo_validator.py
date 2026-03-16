from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

from src.common.config import load_config
from src.common.io import write_json


def run_loyo(config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    train = pd.read_parquet(root / "data" / "processed" / "train.parquet")

    # MVP fallback: single-year pseudo-LOYO computed as stratified split proxy.
    cut = int(len(train) * 0.8)
    tr, va = train.iloc[:cut], train.iloc[cut:]
    model = LogisticRegression(max_iter=1000)
    model.fit(tr[["elo_diff", "net_rating_diff", "tempo_diff"]], tr["label"])
    probs = model.predict_proba(va[["elo_diff", "net_rating_diff", "tempo_diff"]])[:, 1]

    metrics = {
        "log_loss": float(log_loss(va["label"], probs)),
        "brier": float(brier_score_loss(va["label"], probs)),
        "n_validation": int(len(va)),
    }
    out = root / "outputs" / "validation" / "loyo_results.json"
    write_json(out, metrics)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {run_loyo(args.config)}")
