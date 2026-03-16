from __future__ import annotations

import argparse

from src.common.config import load_config
from src.common.io import read_json


def export_bracket_text(year: int, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    selection = read_json(root / "outputs" / f"bracket_{year}_final.json")
    txt = f"""March Madness Office Pool Recommendation ({year})
===============================================
Selected Champion: {selection['selected_champion']}
Champion Probability: {selection['champ_prob']:.3f}
Estimated P(win pool): {selection['p_win_pool']:.3f}
Leverage Score: {selection['leverage_score']:.3f}
Strategy Profile: {selection['selection_logic']}
"""
    out = root / "outputs" / f"bracket_{year}.txt"
    out.write_text(txt, encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {export_bracket_text(args.year, args.config)}")
