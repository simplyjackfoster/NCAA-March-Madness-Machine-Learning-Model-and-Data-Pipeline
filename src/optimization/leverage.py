from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import load_config


def compute_leverage(year: int, config_path: str = "configs/config.yaml"):
    """Pass through pool-simulator p_win_pool from expected_scores; preserve it.

    Reads expected_scores_{year}.parquet (which inherits p_win_pool from candidates
    via compute_expected_scores). Writes leveraged_candidates_{year}.parquet with
    leverage_score column ensured but p_win_pool unchanged.
    """
    cfg = load_config(config_path)
    root = cfg["_root"]
    df = pd.read_parquet(root / "data" / "optimization" / f"expected_scores_{year}.parquet")

    # p_win_pool is already set by pool_simulator and passes through compute_expected_scores.
    # Ensure leverage_score exists as a diagnostic column but do NOT overwrite p_win_pool.
    if "leverage_score" not in df.columns:
        df["leverage_score"] = df["champ_prob"] / df["field_pick"].clip(lower=1e-6)

    out = root / "data" / "optimization" / f"leveraged_candidates_{year}.parquet"
    df.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {compute_leverage(args.year, args.config)}")
