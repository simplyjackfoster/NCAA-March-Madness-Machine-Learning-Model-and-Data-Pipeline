from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.data.source_loader import DataSourceError, load_csv_source


def _default_bracket(num_teams: int) -> pd.DataFrame:
    regions = ["East", "West", "South", "Midwest"]
    rows = []
    for idx in range(num_teams):
        rows.append(
            {
                "team_name": f"Team_{idx + 1:02d}",
                "seed": (idx % 16) + 1,
                "region": regions[(idx // 16) % len(regions)],
                "slot": idx + 1,
            }
        )
    return pd.DataFrame(rows)


def ingest_bracket(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    source_cfg = cfg.get("data_sources", {}).get("bracket", {})

    try:
        df = load_csv_source(
            source_name="bracket",
            root=root,
            year=year,
            local_path=source_cfg.get("local_path"),
            url_template=source_cfg.get("url_template"),
            required_columns=["team_name", "seed", "region"],
        )
    except DataSourceError:
        df = _default_bracket(cfg["data"]["num_teams"])

    if "slot" not in df.columns:
        df = df.copy()
        df["slot"] = range(1, len(df) + 1)

    if "season" not in df.columns:
        df.insert(0, "season", year)
    out_path = root / "data" / "raw" / "bracket" / f"bracket_{year}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["season", "team_name", "seed", "region", "slot"]].to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {ingest_bracket(args.year, args.config)}")
