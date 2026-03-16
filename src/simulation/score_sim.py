from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config


def simulate_scores(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    rng = np.random.default_rng(cfg["data"]["random_seed"] + 17)

    sims = pd.read_parquet(root / "data" / "simulation" / f"score_dist_{year}.parquet")
    adv = pd.read_json(root / "data" / "simulation" / f"advance_probs_{year}.json")
    champ_probs = adv["champion_prob"].to_dict()

    rows = []
    for team, p in champ_probs.items():
        base = 95 * p
        variability = max(5.0, 25 * (1 - p))
        sampled = rng.normal(base, variability, size=2000)
        rows.append(
            {
                "team": team,
                "champion_prob": p,
                "mean_score": float(np.mean(sampled)),
                "p90_score": float(np.quantile(sampled, 0.9)),
                "n_sims": int(len(sims)),
            }
        )

    out = root / "data" / "simulation" / f"team_score_distribution_{year}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("mean_score", ascending=False).to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {simulate_scores(args.year, args.config)}")
