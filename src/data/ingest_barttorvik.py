from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import load_config


def ingest_barttorvik(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    out_path = root / "data" / "raw" / "barttorvik" / f"barttorvik_{year}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Placeholder schema for local/offline MVP.
    teams = [f"Team_{i:02d}" for i in range(1, cfg["data"]["num_teams"] + 1)]
    df = pd.DataFrame(
        {
            "season": year,
            "team_name": teams,
            "adj_o": [105 + (i % 15) for i in range(len(teams))],
            "adj_d": [95 + (i % 12) for i in range(len(teams))],
            "tempo": [65 + (i % 8) for i in range(len(teams))],
        }
    )
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    path = ingest_barttorvik(args.year, args.config)
    print(f"Wrote {path}")
