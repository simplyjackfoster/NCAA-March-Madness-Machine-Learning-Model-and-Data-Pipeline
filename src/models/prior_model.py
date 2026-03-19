from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.common.config import load_config


def train_prior_model(config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    train = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    x = train[["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]]
    y = train["label"]

    model = LogisticRegression(max_iter=1000)
    model.fit(x, y)

    out = root / "artifacts" / "models" / "prior_model.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(model, f)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {train_prior_model(args.config)}")
