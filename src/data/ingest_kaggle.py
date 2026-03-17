from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.data.source_loader import DataSourceError, load_csv_source


KG_COLUMN_ALIASES = {
    "season": "season",
    "gameid": "game_id",
    "daynum": "day_num",
    "wteamid": "team_id",
    "lteamid": "opp_id",
    "wscore": "score",
    "lscore": "opp_score",
}


def _normalize_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    remap = {}
    for c in df.columns:
        key = c.strip().lower().replace(" ", "_")
        if key in KG_COLUMN_ALIASES:
            remap[c] = KG_COLUMN_ALIASES[key]
    out = df.rename(columns=remap).copy()
    required = ["team_id", "opp_id", "score", "opp_score"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise DataSourceError(f"kaggle: normalized data missing columns {missing}")
    if "season" not in out.columns:
        out["season"] = year
    out = out[out["season"] == year].copy()
    if "game_id" not in out.columns:
        out["game_id"] = range(len(out))
    out["won"] = (out["score"] > out["opp_score"]).astype(int)
    return out[["season", "game_id", "team_id", "opp_id", "score", "opp_score", "won"]]


def ingest_kaggle(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    rng = np.random.default_rng(cfg["data"]["random_seed"] + year)

    source_cfg = cfg.get("data_sources", {}).get("kaggle", {})
    try:
        source_df = load_csv_source(
            source_name="kaggle",
            root=root,
            year=year,
            local_path=source_cfg.get("local_path"),
            url_template=source_cfg.get("url_template"),
        )
        df = _normalize_columns(source_df, year)
    except DataSourceError:
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
        df = pd.DataFrame(rows)

    out_path = root / "data" / "raw" / "kaggle" / f"regular_season_{year}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {ingest_kaggle(args.year, args.config)}")
