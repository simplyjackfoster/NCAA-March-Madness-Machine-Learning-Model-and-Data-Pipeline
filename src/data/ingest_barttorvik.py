from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.data.source_loader import DataSourceError, load_csv_source


BT_COLUMN_ALIASES = {
    "team": "team_name",
    "teamname": "team_name",
    "adjoff": "adj_o",
    "adjo": "adj_o",
    "adjde": "adj_d",
    "adjdef": "adj_d",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    remap = {}
    for c in df.columns:
        key = c.strip().lower().replace(" ", "_")
        if key in BT_COLUMN_ALIASES:
            remap[c] = BT_COLUMN_ALIASES[key]
    out = df.rename(columns=remap).copy()
    required = ["team_name", "adj_o", "adj_d"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise DataSourceError(f"barttorvik: normalized data missing columns {missing}")
    if "tempo" not in out.columns:
        out["tempo"] = 68.0  # NCAA average fallback when not in source data
    return out


def ingest_barttorvik(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    out_path = root / "data" / "raw" / "barttorvik" / f"barttorvik_{year}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_cfg = cfg.get("data_sources", {}).get("barttorvik", {})
    df: pd.DataFrame
    try:
        source_df = load_csv_source(
            source_name="barttorvik",
            root=root,
            year=year,
            local_path=source_cfg.get("local_path"),
            url_template=source_cfg.get("url_template"),
        )
        df = _normalize_columns(source_df)
        df = df[["team_name", "adj_o", "adj_d", "tempo"]].copy()
        df.insert(0, "season", year)
    except DataSourceError:
        teams = [f"Team_{i:02d}" for i in range(1, cfg["data"]["num_teams"] + 1)]
        df = pd.DataFrame(
            {
                "season": year,
                "team_name": teams,
                "adj_o": [105 + (i % 15) for i in range(len(teams))],
                "adj_d": [95 + (i % 12) for i in range(len(teams))],
                "tempo": [65 + (i % 8) for i in range(len(teams))],
            }
        )

    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    path = ingest_barttorvik(args.year, args.config)
    print(f"Wrote {path}")
