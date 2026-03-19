from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import pandas as pd

from src.common.config import load_config


def build_ensemble_weights(config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]

    train = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    feature_cols = ["elo_diff", "net_rating_diff", "tempo_diff"]
    # fillna(0) handles constant columns (NaN correlation); clip prevents division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corrs = train[feature_cols].corrwith(train["label"]).fillna(0.0)
    strength = corrs.abs().clip(lower=1e-9)
    total = float(strength.sum())
    weights = {
        "prior": float(strength["elo_diff"] / total),
        "lgbm": float(strength["net_rating_diff"] / total),
        "xgb": float(strength["tempo_diff"] / total),
    }

    out = root / "artifacts" / "models" / "ensemble_weights.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(weights, f)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {build_ensemble_weights(args.config)}")
