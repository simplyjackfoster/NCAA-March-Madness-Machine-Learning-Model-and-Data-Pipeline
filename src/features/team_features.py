from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config


def build_team_features(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    bt_path = root / "data" / "raw" / "barttorvik" / f"barttorvik_{year}.csv"
    cx_path = root / "data" / "crosswalks" / "team_id_map.csv"
    kp_path = root / "data" / "raw" / "kenpom" / f"kenpom_{year}.csv"
    bracket_path = root / "data" / "raw" / "bracket" / f"bracket_{year}.csv"

    bt = pd.read_csv(bt_path)
    cx = pd.read_csv(cx_path)
    teams = bt.merge(cx, left_on="team_name", right_on="barttorvik_name", how="left")
    teams["net_rating"] = teams["adj_o"] - teams["adj_d"]
    teams["elo_pre"] = 1500 + teams["net_rating"] * 8

    # KenPom features (adj_em, luck)
    if kp_path.exists():
        kp = pd.read_csv(kp_path)
        kp_cols = ["team_name"] + [c for c in ["adj_em", "luck"] if c in kp.columns]
        teams = teams.merge(kp[kp_cols], on="team_name", how="left")
    if "adj_em" not in teams.columns:
        teams["adj_em"] = teams["net_rating"]
    if "luck" not in teams.columns:
        teams["luck"] = 0.0
    teams["adj_em"] = teams["adj_em"].fillna(teams["net_rating"])
    teams["luck"] = teams["luck"].fillna(0.0)

    # Massey ordinals (POM, MOR, SAG) for this year
    _MASSEY_SYSTEMS = ["POM", "MOR", "SAG"]
    massey_path = root / "data" / "raw" / "kaggle" / "downloads" / "MMasseyOrdinals.csv"
    if massey_path.exists():
        massey = pd.read_csv(massey_path)
        pre = massey[(massey["Season"] == year) & (massey["RankingDayNum"] <= 133)]
        for system in _MASSEY_SYSTEMS:
            sys_df = (
                pre[pre["SystemName"] == system]
                .sort_values("RankingDayNum")
                .groupby("TeamID")
                .tail(1)[["TeamID", "OrdinalRank"]]
                .rename(columns={"OrdinalRank": f"massey_rank_{system}"})
            )
            teams = teams.merge(
                sys_df.rename(columns={"TeamID": "kaggle_team_id"}),
                on="kaggle_team_id",
                how="left",
            )
    for system in ["POM", "MOR", "SAG"]:
        col = f"massey_rank_{system}"
        if col not in teams.columns:
            teams[col] = np.nan
        # Fill missing with median rank (neutral fallback)
        teams[col] = teams[col].fillna(teams[col].median())

    # Seed from bracket
    if bracket_path.exists():
        bracket = pd.read_csv(bracket_path)
        teams = teams.merge(bracket[["team_name", "seed"]], on="team_name", how="left")
    if "seed" not in teams.columns:
        teams["seed"] = 8
    teams["seed"] = teams["seed"].fillna(8).astype(int)

    out_cols = [
        "season",
        "kaggle_team_id",
        "display_name",
        "adj_o",
        "adj_d",
        "tempo",
        "net_rating",
        "elo_pre",
        "adj_em",
        "luck",
        "seed",
        "massey_rank_POM",
        "massey_rank_MOR",
        "massey_rank_SAG",
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
