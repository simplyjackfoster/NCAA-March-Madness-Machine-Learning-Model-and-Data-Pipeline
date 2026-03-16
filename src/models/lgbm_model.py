from __future__ import annotations

import argparse
import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.common.config import load_config, load_yaml


def train_lgbm_proxy(config_path: str = "configs/config.yaml", params_path: str = "configs/lgbm_params.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    _ = load_yaml(root / params_path)
    train = pd.read_parquet(root / "data" / "processed" / "train.parquet")

    x = train[["elo_diff", "net_rating_diff", "tempo_diff"]]
    y = train["label"]
    model = GradientBoostingClassifier(random_state=cfg["data"]["random_seed"])
    model.fit(x, y)

    out = root / "artifacts" / "models" / "lgbm_model.pkl"
    with out.open("wb") as f:
        pickle.dump(model, f)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--params", default="configs/lgbm_params.yaml")
    args = parser.parse_args()
    print(f"Wrote {train_lgbm_proxy(args.config, args.params)}")
