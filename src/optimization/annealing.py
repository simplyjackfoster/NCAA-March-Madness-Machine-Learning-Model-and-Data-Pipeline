from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import load_config
from src.common.io import write_json


def run_annealing(year: int, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    df = pd.read_parquet(root / "data" / "optimization" / f"leveraged_candidates_{year}.parquet")

    best = df.sort_values(["p_win_pool", "expected_score"], ascending=False).iloc[0]
    out = root / "outputs" / f"annealed_bracket_{year}.json"
    write_json(
        out,
        {
            "year": year,
            "selected_champion": best["champion"],
            "objective": "max p_win_pool then expected_score",
            "p_win_pool": float(best["p_win_pool"]),
            "expected_score": float(best["expected_score"]),
        },
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {run_annealing(args.year, args.config)}")
