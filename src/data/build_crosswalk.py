from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import load_config


def build_crosswalk(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    bt_path = root / "data" / "raw" / "barttorvik" / f"barttorvik_{year}.csv"
    if not bt_path.exists():
        raise FileNotFoundError(f"Missing BartTorvik file: {bt_path}")

    df = pd.read_csv(bt_path)
    out = pd.DataFrame(
        {
            "kaggle_team_id": range(1, len(df) + 1),
            "barttorvik_name": df["team_name"],
            "kenpom_name": df["team_name"],
            "display_name": df["team_name"],
        }
    )
    out_path = root / "data" / "crosswalks" / "team_id_map.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    path = build_crosswalk(args.year, args.config)
    print(f"Wrote {path}")
