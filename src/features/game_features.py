from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config

FEATURES = ["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]
_MASSEY_SYSTEMS = ["POM", "MOR", "SAG"]
_MASSEY_DAY_CUTOFF = 133  # selection Sunday neighborhood


def _load_seed_map(kaggle_dir: Path) -> pd.DataFrame:
    """Returns DataFrame with columns [Season, TeamID, seed_num]."""
    seeds = pd.read_csv(kaggle_dir / "MNCAATourneySeeds.csv")
    seeds["seed_num"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)
    return seeds[["Season", "TeamID", "seed_num"]]


def _load_massey_ranks(kaggle_dir: Path) -> pd.DataFrame:
    """Returns DataFrame with columns [Season, TeamID, rank_POM, rank_MOR, rank_SAG].
    Uses the latest pre-tournament ranking per system per team per season.
    Missing systems get NaN (handled downstream by fill with median).
    """
    massey = pd.read_csv(kaggle_dir / "MMasseyOrdinals.csv")
    pre = massey[massey["RankingDayNum"] <= _MASSEY_DAY_CUTOFF]
    # Keep latest ranking day per system/team/season
    latest = (
        pre.sort_values("RankingDayNum")
        .groupby(["Season", "SystemName", "TeamID"])
        .tail(1)
    )
    result = None
    for system in _MASSEY_SYSTEMS:
        sys_df = (
            latest[latest["SystemName"] == system][["Season", "TeamID", "OrdinalRank"]]
            .rename(columns={"OrdinalRank": f"rank_{system}"})
        )
        if result is None:
            result = sys_df
        else:
            result = result.merge(sys_df, on=["Season", "TeamID"], how="outer")
    return result if result is not None else pd.DataFrame()


def build_game_features(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    kaggle_dir = root / "data" / "raw" / "kaggle" / "downloads"

    results = pd.read_csv(kaggle_dir / "MNCAATourneyCompactResults.csv")
    seed_map = _load_seed_map(kaggle_dir)
    rank_map = _load_massey_ranks(kaggle_dir)

    # Build one row per game from winner's perspective, then mirror
    rows = []
    for _, game in results.iterrows():
        season = int(game["Season"])
        w, l = int(game["WTeamID"]), int(game["LTeamID"])

        def get_seed(tid):
            s = seed_map[(seed_map["Season"] == season) & (seed_map["TeamID"] == tid)]
            return int(s["seed_num"].iloc[0]) if len(s) else 8

        def get_rank(tid, system):
            if rank_map is None or rank_map.empty:
                return np.nan
            col = f"rank_{system}"
            if col not in rank_map.columns:
                return np.nan
            r = rank_map[(rank_map["Season"] == season) & (rank_map["TeamID"] == tid)]
            return float(r[col].iloc[0]) if len(r) else np.nan

        ws, ls = get_seed(w), get_seed(l)
        row_w = {
            "season": season,
            "team_id": w,
            "opp_id": l,
            "seed_diff": ws - ls,
            "label": 1,
        }
        row_l = {
            "season": season,
            "team_id": l,
            "opp_id": w,
            "seed_diff": ls - ws,
            "label": 0,
        }
        for system in _MASSEY_SYSTEMS:
            wr = get_rank(w, system)
            lr = get_rank(l, system)
            row_w[f"rank_diff_{system}"] = wr - lr if not (np.isnan(wr) or np.isnan(lr)) else np.nan
            row_l[f"rank_diff_{system}"] = lr - wr if not (np.isnan(wr) or np.isnan(lr)) else np.nan
        rows.append(row_w)
        rows.append(row_l)

    df = pd.DataFrame(rows)

    # Fill missing Massey ranks with median (teams not covered by a system)
    for col in [f"rank_diff_{s}" for s in _MASSEY_SYSTEMS]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    out_path = root / "data" / "features" / f"games_{year}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / "data" / "processed" / "train.parquet", index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {build_game_features(args.year, args.config)}")
