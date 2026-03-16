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
    # Simple deterministic weighting by feature signal strength.
    strength = abs(train[["elo_diff", "net_rating_diff", "tempo_diff"]].corrwith(train["label"]))
    total = float(strength.sum()) + 1e-9
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
