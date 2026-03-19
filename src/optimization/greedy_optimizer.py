from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import load_config
from src.common.io import write_json


def select_final_bracket(year: int, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    df = pd.read_parquet(root / "data" / "optimization" / f"leveraged_candidates_{year}.parquet")
    min_champ_prob = float(cfg["optimization"]["min_champion_prob"])
    risk = cfg["optimization"]["risk_tolerance"]
    pool_size = int(cfg["optimization"]["pool_size"])

    viable = df[df["champ_prob"] >= min_champ_prob].sort_values("p_win_pool", ascending=False)
    if viable.empty:
        viable = df.sort_values("p_win_pool", ascending=False)

    if risk == "conservative":
        final = viable.head(3).sort_values("expected_score", ascending=False).iloc[0]
    elif risk == "balanced":
        final = viable.iloc[0]
    else:
        baseline = 1.0 / max(pool_size, 1)
        high_lev = viable[viable["p_win_pool"] > 1.5 * baseline]
        final = (high_lev if not high_lev.empty else viable).sort_values("leverage_score", ascending=False).iloc[0]

    out_json = root / "outputs" / f"bracket_{year}_final.json"
    out_csv = root / "outputs" / f"pareto_{year}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    viable.to_csv(out_csv, index=False)

    top3 = viable.head(3)
    top_picks = [
        {
            "champion": row["champion"],
            "model_prob": float(row["champ_prob"]),
            "field_pick": float(row["field_pick"]),
            "p_win_pool": float(row["p_win_pool"]),
        }
        for _, row in top3.iterrows()
    ]

    write_json(
        out_json,
        {
            "year": year,
            "selection_logic": risk,
            "selected_champion": final["champion"],
            "champ_prob": float(final["champ_prob"]),
            "p_win_pool": float(final["p_win_pool"]),
            "leverage_score": float(final["leverage_score"]),
            "top_picks": top_picks,
        },
    )
    return out_json, out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    a, b = select_final_bracket(args.year, args.config)
    print(f"Wrote {a}\nWrote {b}")
