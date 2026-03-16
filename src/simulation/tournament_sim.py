from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.io import write_json


def run_simulation(year: int, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    rng = np.random.default_rng(cfg["data"]["random_seed"])
    mat = np.load(root / "data" / "tournament" / f"prob_matrix_{year}.npy")
    teams = pd.read_parquet(root / "data" / "features" / f"team_season_{year}.parquet")
    n = len(teams)
    sims = int(cfg["simulation"]["num_sims"])

    champions = Counter()
    rows = []
    for sim in range(sims):
        alive = list(range(n))
        round_num = 0
        while len(alive) > 1:
            round_num += 1
            nxt = []
            for i in range(0, len(alive), 2):
                a, b = alive[i], alive[i + 1]
                winner = a if rng.random() < mat[a, b] else b
                nxt.append(winner)
            alive = nxt
        champ = alive[0]
        champions[int(champ)] += 1
        rows.append({"sim_id": sim, "champion_idx": int(champ), "champion": teams.iloc[champ]["display_name"]})

    champ_probs = {
        teams.iloc[idx]["display_name"]: cnt / sims for idx, cnt in champions.items()
    }
    adv_path = root / "data" / "simulation" / f"advance_probs_{year}.json"
    write_json(adv_path, {"champion_prob": champ_probs})

    dist_df = pd.DataFrame(rows)
    dist_path = root / "data" / "simulation" / f"score_dist_{year}.parquet"
    dist_path.parent.mkdir(parents=True, exist_ok=True)
    dist_df.to_parquet(dist_path, index=False)
    return adv_path, dist_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    a, b = run_simulation(args.year, args.config)
    print(f"Wrote {a}\nWrote {b}")
