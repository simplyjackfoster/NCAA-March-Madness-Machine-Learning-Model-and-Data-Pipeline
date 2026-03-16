from __future__ import annotations

import argparse

from src.common.config import load_config
from src.common.io import read_json, write_json
from src.field.espn_loader import load_espn_pick_rates


def build_pool_model(year: int, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    adv = read_json(root / "data" / "simulation" / f"advance_probs_{year}.json")
    champ = adv["champion_prob"]

    espn_path = load_espn_pick_rates(year, config_path)
    espn = read_json(espn_path)

    pool = {
        t: {
            "model_champion_prob": champ[t],
            "espn_pick_pct": espn[t],
            "leverage": champ[t] / max(espn[t], 1e-6),
        }
        for t in champ
    }
    pool_path = root / "data" / "field" / f"pool_picks_{year}.json"
    write_json(pool_path, pool)
    return espn_path, pool_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    a, b = build_pool_model(args.year, args.config)
    print(f"Wrote {a}\nWrote {b}")
