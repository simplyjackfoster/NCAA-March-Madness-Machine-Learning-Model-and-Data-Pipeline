from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config


def ingest_kaggle(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    rng = np.random.default_rng(cfg["data"]["random_seed"] + year)

    n_games = max(1500, cfg["data"]["num_teams"] * 30)
    team_ids = np.arange(1, cfg["data"]["num_teams"] + 1)
    rows = []
    for game_id in range(n_games):
        t1, t2 = rng.choice(team_ids, size=2, replace=False)
        base = rng.normal(0, 1)
        score_t1 = int(65 + base * 7 + rng.normal(0, 9))
        score_t2 = int(65 - base * 7 + rng.normal(0, 9))
        rows.append({
            "season": year,
            "game_id": game_id,
            "team_id": int(t1),
            "opp_id": int(t2),
            "score": score_t1,
            "opp_score": score_t2,
            "won": int(score_t1 > score_t2),
        })

    out_path = root / "data" / "raw" / "kaggle" / f"regular_season_{year}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {ingest_kaggle(args.year, args.config)}")
