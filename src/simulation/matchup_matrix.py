from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from src.common.config import load_config


def build_matchup_matrix(year: int, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    team_df = pd.read_parquet(root / "data" / "features" / f"team_season_{year}.parquet")
    with (root / "artifacts" / "models" / "prior_model.pkl").open("rb") as f:
        model = pickle.load(f)

    with (root / "artifacts" / "calibrators" / "isotonic.pkl").open("rb") as f:
        calibrator = pickle.load(f)

    FEATURES = ["elo_diff", "net_rating_diff", "tempo_diff"]
    n = len(team_df)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = team_df.iloc[i]
            b = team_df.iloc[j]
            x = pd.DataFrame(
                [[
                    a["elo_pre"] - b["elo_pre"],
                    a["net_rating"] - b["net_rating"],
                    a["tempo"] - b["tempo"],
                ]],
                columns=FEATURES,
            )
            raw_prob = float(model.predict_proba(x)[0, 1])
            cal_input = np.array([[raw_prob]])
            mat[i, j] = float(calibrator.predict_proba(cal_input)[0, 1])

    out = root / "data" / "tournament" / f"prob_matrix_{year}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, mat)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {build_matchup_matrix(args.year, args.config)}")
