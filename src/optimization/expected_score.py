from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import load_config


def compute_expected_scores(year: int, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    candidates = pd.read_parquet(root / "data" / "optimization" / f"candidates_{year}.parquet")
    score_dist = pd.read_parquet(root / "data" / "simulation" / f"team_score_distribution_{year}.parquet")

    merged = candidates.merge(score_dist[["team", "mean_score", "p90_score"]], left_on="champion", right_on="team", how="left")
    merged["expected_score"] = merged["mean_score"].fillna(merged["expected_score"])

    out = root / "data" / "optimization" / f"expected_scores_{year}.parquet"
    merged.drop(columns=["team"]).to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {compute_expected_scores(args.year, args.config)}")
