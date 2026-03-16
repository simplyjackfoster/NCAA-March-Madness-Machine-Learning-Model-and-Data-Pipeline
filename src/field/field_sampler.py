from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.io import read_json


def sample_field_brackets(year: int, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    rng = np.random.default_rng(cfg["data"]["random_seed"] + 99)
    pool_size = int(cfg["optimization"]["pool_size"])

    pool = read_json(root / "data" / "field" / f"pool_picks_{year}.json")
    teams = list(pool.keys())
    probs = np.array([pool[t]["espn_pick_pct"] for t in teams], dtype=float)
    probs /= probs.sum()

    champs = rng.choice(teams, p=probs, size=pool_size, replace=True)
    out = root / "data" / "field" / f"sampled_field_{year}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"entry_id": range(pool_size), "champion_pick": champs}).to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {sample_field_brackets(args.year, args.config)}")
