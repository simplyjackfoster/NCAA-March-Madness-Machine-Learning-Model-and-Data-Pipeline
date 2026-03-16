from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.io import read_json


def export_strategy_report(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]

    final = read_json(root / "outputs" / f"bracket_{year}_final.json")
    candidates = pd.read_parquet(root / "data" / "optimization" / f"candidates_{year}.parquet")

    top = candidates.sort_values("p_win_pool", ascending=False).head(10)
    lines = [
        f"Strategy Report - {year}",
        f"Selected champion: {final['selected_champion']}",
        f"Risk profile: {final['selection_logic']}",
        "",
        "Top alternatives:",
    ]

    for _, row in top.iterrows():
        lines.append(
            f"- {row['champion']}: champ_prob={row['champ_prob']:.3f}, "
            f"field_pick={row['field_pick']:.3f}, p_win_pool={row['p_win_pool']:.3f}"
        )

    out = root / "outputs" / f"strategy_report_{year}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {export_strategy_report(args.year, args.config)}")
