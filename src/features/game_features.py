from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config


def build_game_features(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    rng = np.random.default_rng(cfg["data"]["random_seed"])
    root = cfg["_root"]
    team_path = root / "data" / "features" / f"team_season_{year}.parquet"
    teams = pd.read_parquet(team_path)

    games = []
    team_ids = teams["kaggle_team_id"].tolist()
    for _ in range(max(1200, len(team_ids) * 20)):
        t1, t2 = rng.choice(team_ids, size=2, replace=False)
        a = teams.loc[teams["kaggle_team_id"] == t1].iloc[0]
        b = teams.loc[teams["kaggle_team_id"] == t2].iloc[0]
        diff = a["net_rating"] - b["net_rating"]
        games.append(
            {
                "season": year,
                "team_id": int(t1),
                "opp_id": int(t2),
                "net_rating_diff": diff,
                "elo_diff": a["elo_pre"] - b["elo_pre"],
                "tempo_diff": a["tempo"] - b["tempo"],
                "seed_diff": int(a.get("seed", 8)) - int(b.get("seed", 8)),
                "label": int(diff + rng.normal(0, 2) > 0),
            }
        )

    df = pd.DataFrame(games)
    out_path = root / "data" / "features" / f"games_{year}.parquet"
    df.to_parquet(out_path, index=False)

    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / "data" / "processed" / "train.parquet", index=False)
    df.head(256).to_parquet(root / "data" / "processed" / "tourney.parquet", index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {build_game_features(args.year, args.config)}")
