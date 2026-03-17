"""Download Kaggle March Machine Learning Mania dataset and prepare for pipeline.

Usage:
    python scripts/download_data.py [--year YEAR] [--seasons 2016 2017 ... 2026]

Requires ~/.kaggle/kaggle.json with your Kaggle API credentials.
See SETUP.md for instructions.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Union

import pandas as pd

COMPETITION = "march-machine-learning-mania-2026"
RESULTS_FILE = "MRegularSeasonCompactResults.csv"
TEAMS_FILE = "MTeams.csv"

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_RAW = ROOT / "data" / "raw" / "kaggle"


def _check_credentials() -> None:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            f"Kaggle credentials not found at {kaggle_json}.\n"
            "See SETUP.md for instructions on creating your API token."
        )


def _download_competition(download_dir: Path) -> Path:
    """Download competition zip via Kaggle API and return the extracted directory."""
    import kaggle  # noqa: PLC0415

    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {COMPETITION} to {download_dir} ...")
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(COMPETITION, path=str(download_dir), quiet=False)

    zips = list(download_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No zip file found in {download_dir} after download.")
    zip_path = zips[0]
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(download_dir)

    return download_dir


def process_game_results(
    results_csv: Path,
    season: Union[int, list[int]],
) -> pd.DataFrame:
    """Load and filter game results by season(s). Returns raw DataFrame."""
    df = pd.read_csv(results_csv)
    seasons = [season] if isinstance(season, int) else list(season)
    return df[df["Season"].isin(seasons)].reset_index(drop=True)


def process_teams(teams_csv: Path) -> pd.DataFrame:
    """Load teams lookup. Returns DataFrame with only TeamID and TeamName columns."""
    return pd.read_csv(teams_csv)[["TeamID", "TeamName"]].copy()


def main(year: int, seasons: list[int]) -> None:
    _check_credentials()

    download_dir = KAGGLE_RAW / "downloads"
    extracted = _download_competition(download_dir)

    results_csv = extracted / RESULTS_FILE
    teams_csv = extracted / TEAMS_FILE

    if not results_csv.exists():
        files = [f.name for f in extracted.iterdir()]
        raise FileNotFoundError(
            f"{RESULTS_FILE} not found in {extracted}.\nFiles present: {files}"
        )
    if not teams_csv.exists():
        files = [f.name for f in extracted.iterdir()]
        raise FileNotFoundError(
            f"{TEAMS_FILE} not found in {extracted}.\nFiles present: {files}"
        )

    # Save MTeams.csv for crosswalk to use
    teams_out = KAGGLE_RAW / "MTeams.csv"
    process_teams(teams_csv).to_csv(teams_out, index=False)
    print(f"Saved team lookup → {teams_out}")

    # Filter game results and save
    results = process_game_results(results_csv, season=seasons)
    out_path = KAGGLE_RAW / f"regular_season_{year}.csv"
    results.to_csv(out_path, index=False)
    print(f"Wrote game results → {out_path} ({len(results)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(range(2016, 2027)),
        help="Seasons to include for model training (default: 2016–2026)",
    )
    args = parser.parse_args()
    main(year=args.year, seasons=args.seasons)
