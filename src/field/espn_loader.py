from __future__ import annotations

import argparse
from pathlib import Path

from src.common.config import load_config
from src.common.io import read_json, write_json


def load_espn_pick_rates(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]

    adv = read_json(root / "data" / "simulation" / f"advance_probs_{year}.json")
    champ = adv["champion_prob"]

    # Offline default proxy for ESPN pick rates.
    picks = {team: min(0.85, p * 1.3 + 0.01) for team, p in champ.items()}
    total = sum(picks.values()) or 1.0
    picks = {team: value / total for team, value in picks.items()}

    out = root / "data" / "field" / f"espn_picks_{year}.json"
    write_json(out, picks)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {load_espn_pick_rates(args.year, args.config)}")
