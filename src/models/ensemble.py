from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.common.config import load_config


def build_ensemble_weights(config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]

    train = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    feature_cols = ["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]
    # Only correlate non-zero-variance features that exist in the data
    non_zero_var_cols = [col for col in feature_cols if col in train.columns and train[col].var() > 0]
    corrs = train[non_zero_var_cols].corrwith(train["label"]).reindex(feature_cols).fillna(0.0)
    strength = corrs.abs().clip(lower=1e-9)
    total = float(strength.sum())
    weights = {
        "prior": float(strength["seed_diff"] / total),
        "lgbm": float(strength["rank_diff_POM"] / total),
        "xgb": float(strength["rank_diff_MOR"] / total),
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
