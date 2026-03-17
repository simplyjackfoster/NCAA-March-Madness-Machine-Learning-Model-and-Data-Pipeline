from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.io import read_json


def _build_full_bracket(year: int, root) -> str:
    mat = np.load(root / "data" / "tournament" / f"prob_matrix_{year}.npy")
    teams = pd.read_parquet(root / "data" / "features" / f"team_season_{year}.parquet")
    bracket = pd.read_csv(root / "data" / "raw" / "bracket" / f"bracket_{year}.csv")

    merged = bracket.merge(
        teams.reset_index().rename(columns={"index": "team_idx"}),
        left_on="team_name", right_on="display_name", how="left",
    )
    ordered = (
        merged.sort_values(["slot", "seed"], ascending=[True, True])
        .dropna(subset=["team_idx"])
    )
    idx_to_name = dict(zip(ordered["team_idx"].astype(int), ordered["team_name"]))
    idx_to_seed = dict(zip(ordered["team_idx"].astype(int), ordered["seed"].astype(int)))
    alive = ordered["team_idx"].astype(int).tolist()

    round_names = ["First Round", "Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    lines = []
    for rname in round_names:
        lines.append(f"--- {rname} ---")
        nxt = []
        for i in range(0, len(alive), 2):
            a, b = alive[i], alive[i + 1]
            prob_a = mat[a, b]
            winner = a if prob_a >= 0.5 else b
            win_prob = max(prob_a, 1 - prob_a)
            lines.append(
                f"  ({idx_to_seed[a]}) {idx_to_name[a]:<22} vs "
                f"({idx_to_seed[b]}) {idx_to_name[b]:<22}  ->  "
                f"({idx_to_seed[winner]}) {idx_to_name[winner]} [{win_prob:.0%}]"
            )
            nxt.append(winner)
        alive = nxt
        lines.append("")
    return "\n".join(lines)


def export_bracket_text(year: int, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    selection = read_json(root / "outputs" / f"bracket_{year}_final.json")

    summary = f"""March Madness Office Pool Recommendation ({year})
===============================================
Selected Champion: {selection['selected_champion']}
Champion Probability: {selection['champ_prob']:.3f}
Estimated P(win pool): {selection['p_win_pool']:.3f}
Leverage Score: {selection['leverage_score']:.3f}
Strategy Profile: {selection['selection_logic']}
"""
    out = root / "outputs" / f"bracket_{year}.txt"
    out.write_text(summary, encoding="utf-8")

    try:
        full = _build_full_bracket(year, root)
        full_txt = f"{summary}\n{full}"
        full_out = root / "outputs" / f"full_bracket_{year}.txt"
        full_out.write_text(full_txt, encoding="utf-8")
    except Exception:
        pass  # don't fail the pipeline if full bracket can't be built

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {export_bracket_text(args.year, args.config)}")
