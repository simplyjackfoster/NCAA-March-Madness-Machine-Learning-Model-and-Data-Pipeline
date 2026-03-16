from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import load_config


def ingest_kenpom(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]

    teams = [f"Team_{i:02d}" for i in range(1, cfg["data"]["num_teams"] + 1)]
    df = pd.DataFrame(
        {
            "season": year,
            "team_name": teams,
            "adj_em": [8 + (i % 18) for i in range(len(teams))],
            "adj_t": [64 + (i % 9) for i in range(len(teams))],
            "luck": [-0.02 + (i % 7) * 0.01 for i in range(len(teams))],
        }
    )

    out_path = root / "data" / "raw" / "kenpom" / f"kenpom_{year}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {ingest_kenpom(args.year, args.config)}")
