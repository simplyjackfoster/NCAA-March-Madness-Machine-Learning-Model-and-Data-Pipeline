from __future__ import annotations

import argparse
import difflib
from pathlib import Path

import pandas as pd

from src.common.config import load_config


def _match_names_to_kaggle_ids(bt_names: list[str], kaggle_df: pd.DataFrame) -> list[int]:
    """Fuzzy-match Barttorvik team names to Kaggle TeamNames. Returns list of TeamIDs.

    Uses difflib.get_close_matches with cutoff=0.6. Unmatched names get -1.
    """
    kaggle_names = kaggle_df["TeamName"].tolist()
    id_by_name = dict(zip(kaggle_df["TeamName"], kaggle_df["TeamID"]))
    result = []
    for name in bt_names:
        matches = difflib.get_close_matches(name, kaggle_names, n=1, cutoff=0.6)
        result.append(id_by_name[matches[0]] if matches else -1)
    return result


def build_crosswalk(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]

    # Read barttorvik path from config if set, otherwise use default location
    bt_local = (cfg.get("data_sources", {}).get("barttorvik") or {}).get("local_path")
    bt_path = Path(bt_local) if bt_local else root / "data" / "raw" / "barttorvik" / f"barttorvik_{year}.csv"
    if not bt_path.exists():
        raise FileNotFoundError(f"Missing BartTorvik file: {bt_path}")

    bt_df = pd.read_csv(bt_path)
    bt_names = bt_df["team_name"].tolist()

    # Read MTeams path from config if set, otherwise use default location
    mteams_local = (cfg.get("crosswalk") or {}).get("mteams_path")
    mteams_path = Path(mteams_local) if mteams_local else root / "data" / "raw" / "kaggle" / "MTeams.csv"

    if mteams_path.exists():
        kaggle_df = pd.read_csv(mteams_path)
        kaggle_ids = _match_names_to_kaggle_ids(bt_names, kaggle_df)
    else:
        # Synthetic data mode: sequential IDs
        kaggle_ids = list(range(1, len(bt_names) + 1))

    out = pd.DataFrame(
        {
            "kaggle_team_id": kaggle_ids,
            "barttorvik_name": bt_names,
            "kenpom_name": bt_names,
            "display_name": bt_names,
        }
    )
    out_path = root / "data" / "crosswalks" / "team_id_map.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    path = build_crosswalk(args.year, args.config)
    print(f"Wrote {path}")
