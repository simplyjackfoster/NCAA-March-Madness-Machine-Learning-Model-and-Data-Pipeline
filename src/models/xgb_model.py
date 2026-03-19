from __future__ import annotations

import argparse
import pickle

import pandas as pd
from xgboost import XGBClassifier

from src.common.config import load_config, load_yaml


def train_xgb_model(config_path: str = "configs/config.yaml", params_path: str = "configs/xgb_params.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    params = load_yaml(root / params_path)
    train = pd.read_parquet(root / "data" / "processed" / "train.parquet")

    x = train[["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]]
    y = train["label"]

    params.setdefault("random_state", cfg["data"]["random_seed"])
    model = XGBClassifier(**params)
    model.fit(x, y)

    out = root / "artifacts" / "models" / "xgb_model.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(model, f)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--params", default="configs/xgb_params.yaml")
    args = parser.parse_args()
    print(f"Wrote {train_xgb_model(args.config, args.params)}")
