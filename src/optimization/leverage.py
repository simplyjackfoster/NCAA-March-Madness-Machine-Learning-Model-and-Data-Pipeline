from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import load_config


def compute_leverage(year: int, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    df = pd.read_parquet(root / "data" / "optimization" / f"expected_scores_{year}.parquet")
    df["leverage_score"] = df["champ_prob"] / df["field_pick"].clip(lower=1e-6)
    df["p_win_pool"] = df["champ_prob"] * df["leverage_score"]

    out = root / "data" / "optimization" / f"leveraged_candidates_{year}.parquet"
    df.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {compute_leverage(args.year, args.config)}")
