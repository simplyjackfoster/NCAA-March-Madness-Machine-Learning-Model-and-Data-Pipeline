from __future__ import annotations

import argparse
import pickle

import pandas as pd
from sklearn.calibration import IsotonicRegression

from src.common.config import load_config
from src.models.diagnostics import write_calibration_diagnostics


def train_calibrator(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    train = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    probs = (train["elo_diff"].to_numpy() - train["elo_diff"].min())
    probs = probs / (probs.max() + 1e-9)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, train["label"])

    out = root / "artifacts" / "calibrators" / "isotonic.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(iso, f)

    write_calibration_diagnostics(train["label"].to_numpy(), iso.predict(probs), root / "outputs" / "validation")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {train_calibrator(args.config)}")
