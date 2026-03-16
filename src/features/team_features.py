from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import load_config


def build_team_features(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    bt_path = root / "data" / "raw" / "barttorvik" / f"barttorvik_{year}.csv"
    cx_path = root / "data" / "crosswalks" / "team_id_map.csv"

    bt = pd.read_csv(bt_path)
    cx = pd.read_csv(cx_path)
    teams = bt.merge(cx, left_on="team_name", right_on="barttorvik_name", how="left")
    teams["net_rating"] = teams["adj_o"] - teams["adj_d"]
    teams["elo_pre"] = 1500 + teams["net_rating"] * 8

    out_cols = [
        "season",
        "kaggle_team_id",
        "display_name",
        "adj_o",
        "adj_d",
        "tempo",
        "net_rating",
        "elo_pre",
    ]
    out_path = root / "data" / "features" / f"team_season_{year}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    teams[out_cols].to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {build_team_features(args.year, args.config)}")
