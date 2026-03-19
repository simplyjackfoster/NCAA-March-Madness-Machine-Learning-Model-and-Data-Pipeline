from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.io import write_json
from src.field.seed_popularity import get_seed_popularity


def load_espn_pick_rates(year: int, config_path: str = "configs/config.yaml") -> Path:
    """Compute field championship pick rates using historical seed popularity.

    Returns a JSON file mapping team_name -> pick_pct (sums to 1.0).
    """
    cfg = load_config(config_path)
    root = cfg["_root"]
    bracket_path = root / "data" / "raw" / "bracket" / f"bracket_{year}.csv"

    bracket = pd.read_csv(bracket_path)
    picks = get_seed_popularity(bracket)

    out = root / "data" / "field" / f"espn_picks_{year}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, picks)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {load_espn_pick_rates(args.year, args.config)}")
