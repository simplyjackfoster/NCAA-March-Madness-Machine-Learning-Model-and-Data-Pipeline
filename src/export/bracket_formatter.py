from __future__ import annotations

import argparse
from datetime import datetime

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
    seed_col = "seed_x" if "seed_x" in merged.columns else "seed"
    ordered = (
        merged.sort_values(["slot", seed_col], ascending=[True, True])
        .dropna(subset=["team_idx"])
    )
    idx_to_name = dict(zip(ordered["team_idx"].astype(int), ordered["team_name"]))
    idx_to_seed = dict(zip(ordered["team_idx"].astype(int), ordered[seed_col].astype(int)))
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

    # Build top-3 picks header (prepended to full bracket only, not to summary .txt)
    top_picks_header = ""
    if "top_picks" in selection:
        lines_list = [
            f"Top Champion Picks for Pool (pool size: {cfg['optimization']['pool_size']})",
            "=" * 50,
        ]
        for rank, pick in enumerate(selection["top_picks"], 1):
            lines_list.append(
                f"{rank}. {pick['champion']:<22} "
                f"model={pick['model_prob']:.0%}  "
                f"field={pick['field_pick']:.0%}  "
                f"P(win pool)={pick['p_win_pool']:.1%}"
            )
        top_picks_header = "\n".join(lines_list) + "\n\n"

    try:
        full = _build_full_bracket(year, root)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_txt = f"{top_picks_header}{summary}\n{full}"
        # Always overwrite the latest for easy access
        latest = root / "outputs" / f"full_bracket_{year}.txt"
        latest.write_text(full_txt, encoding="utf-8")
        # Also save a timestamped copy to preserve each run
        runs_dir = root / "outputs" / "runs"
        runs_dir.mkdir(exist_ok=True)
        (runs_dir / f"full_bracket_{year}_{ts}.txt").write_text(full_txt, encoding="utf-8")
    except Exception:
        pass  # don't fail the pipeline if full bracket can't be built

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {export_bracket_text(args.year, args.config)}")
