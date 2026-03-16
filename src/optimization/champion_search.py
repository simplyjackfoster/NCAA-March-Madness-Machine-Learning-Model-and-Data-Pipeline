from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import load_config
from src.common.io import read_json


def run_champion_search(year: int, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    pool = read_json(root / "data" / "field" / f"pool_picks_{year}.json")

    rows = []
    for team, rec in pool.items():
        p = rec["model_champion_prob"]
        field_p = rec["espn_pick_pct"]
        leverage = rec["leverage"]
        rows.append(
            {
                "champion": team,
                "champ_prob": p,
                "field_pick": field_p,
                "leverage_score": leverage,
                "expected_score": 192 * p,
                "p_win_pool": p * max(leverage, 0.01),
            }
        )

    df = pd.DataFrame(rows).sort_values("p_win_pool", ascending=False)
    out = root / "data" / "optimization" / f"candidates_{year}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {run_champion_search(args.year, args.config)}")
