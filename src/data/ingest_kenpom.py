from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.data.source_loader import DataSourceError, load_csv_source


KP_COLUMN_ALIASES = {
    "team": "team_name",
    "teamname": "team_name",
    "adjem": "adj_em",
    "adjt": "adj_t",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    remap = {}
    for c in df.columns:
        key = c.strip().lower().replace(" ", "_")
        if key in KP_COLUMN_ALIASES:
            remap[c] = KP_COLUMN_ALIASES[key]
    out = df.rename(columns=remap).copy()
    required = ["team_name", "adj_em", "adj_t", "luck"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise DataSourceError(f"kenpom: normalized data missing columns {missing}")
    return out


def ingest_kenpom(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]

    source_cfg = cfg.get("data_sources", {}).get("kenpom", {})
    try:
        source_df = load_csv_source(
            source_name="kenpom",
            root=root,
            year=year,
            local_path=source_cfg.get("local_path"),
            url_template=source_cfg.get("url_template"),
        )
        df = _normalize_columns(source_df)
        df = df[["team_name", "adj_em", "adj_t", "luck"]].copy()
        df.insert(0, "season", year)
    except DataSourceError:
        teams = [f"Team_{i:02d}" for i in range(1, cfg["data"]["num_teams"] + 1)]
        df = pd.DataFrame(
            {
                "season": year,
                "team_name": teams,
                "adj_em": [8 + (i % 18) for i in range(len(teams))],
                "adj_t": [64 + (i % 9) for i in range(len(teams))],
                "luck": [-0.02 + (i % 7) * 0.01 for i in range(len(teams))],
            }
        )

    out_path = root / "data" / "raw" / "kenpom" / f"kenpom_{year}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {ingest_kenpom(args.year, args.config)}")
