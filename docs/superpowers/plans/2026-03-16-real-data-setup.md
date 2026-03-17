# Real Data Setup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing March Madness pipeline to real 2026 data — Kaggle historical games (automated download), Barttorvik + KenPom (manual CSV paste), and the 2026 bracket (pre-populated from PDF).

**Architecture:** Static data files + config changes handle the bracket; a new `scripts/download_data.py` handles Kaggle; `src/data/build_crosswalk.py` is fixed to map real Kaggle team IDs via fuzzy name matching instead of fake sequential ones. Barttorvik/KenPom configs stay `null` until the user pastes data, at which point they update two lines in `config.yaml` — this avoids silently ingesting empty CSVs as real data.

**Tech Stack:** Python 3.9+, pandas, kaggle Python package, difflib (stdlib), PyYAML, pytest

---

## Chunk 1: Static files, config, and dependencies

### Task 1: Add `kaggle` to requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add kaggle package**

In `requirements.txt`, the file currently ends with `pyarrow>=15.0` followed by a blank line. Replace the trailing blank line so the file ends with:
```
numpy>=1.26
pandas>=2.2
pyyaml>=6.0
scikit-learn>=1.4
lightgbm>=4.3
xgboost>=2.0
pyarrow>=15.0
kaggle>=1.6
```

- [ ] **Step 2: Verify install**

```bash
pip install -r requirements.txt
```
Expected: installs without error, `kaggle` CLI is available.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add kaggle package to requirements"
```

---

### Task 2: Update `configs/config.yaml`

The bracket and Kaggle paths are set now (those files are always present after Task 4 and `download_data.py`). Barttorvik and KenPom paths stay `null` — the user will uncomment them after pasting their stats.

**Files:**
- Modify: `configs/config.yaml`

- [ ] **Step 1: Update config**

Replace the entire file contents with:

```yaml
project:
  target_year: 2026
  base_data_dir: data
  artifacts_dir: artifacts
  outputs_dir: outputs

data:
  random_seed: 42
  num_teams: 64

# External data connectors.
# bracket and kaggle are set automatically — do not change these.
# After pasting Barttorvik and KenPom stats, uncomment those local_path lines.
data_sources:
  barttorvik:
    local_path:       # uncomment and set after pasting: data/raw/barttorvik/barttorvik_2026.csv
    url_template:
  kenpom:
    local_path:       # uncomment and set after pasting: data/raw/kenpom/kenpom_2026.csv
    url_template:
  kaggle:
    local_path: data/raw/kaggle/regular_season_2026.csv
    url_template:
  bracket:
    local_path: data/raw/bracket/bracket_2026.csv
    url_template:

simulation:
  num_sims: 5000

optimization:
  min_champion_prob: 0.02
  risk_tolerance: balanced
  pool_size: 50
```

- [ ] **Step 2: Commit**

```bash
git add configs/config.yaml
git commit -m "config: wire bracket and kaggle paths; annotate barttorvik/kenpom for manual activation"
```

---

### Task 3: Create SETUP.md

**Files:**
- Create: `SETUP.md`

- [ ] **Step 1: Write SETUP.md**

Create `SETUP.md` at the project root with this content:

```markdown
# Setup Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Kaggle API Credentials (one-time)

1. Log in to [kaggle.com](https://www.kaggle.com)
2. Profile → Settings → API → **Create New Token** — downloads `kaggle.json`
3. Move it: `mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## 3. Download Kaggle Data

```bash
python scripts/download_data.py
```

Downloads the March Machine Learning Mania dataset and writes:
- `data/raw/kaggle/regular_season_2026.csv` — historical game results for model training
- `data/raw/kaggle/MTeams.csv` — team ID lookup used by the crosswalk

## 4. Paste Barttorvik Team Stats

1. Go to [barttorvik.com](https://barttorvik.com) → Team Stats
2. Copy the table for the current season
3. Paste into `data/raw/barttorvik/barttorvik_2026.csv`
   - Required columns: `team_name`, `adj_o`, `adj_d`, `tempo`
   - Other columns are ignored
4. In `configs/config.yaml`, set:
   ```yaml
   barttorvik:
     local_path: data/raw/barttorvik/barttorvik_2026.csv
   ```

## 5. Paste KenPom Stats

1. Go to [kenpom.com](https://kenpom.com) → main ratings table
2. Copy and paste into `data/raw/kenpom/kenpom_2026.csv`
   - Required columns: `team_name`, `adj_em`, `adj_t`, `luck`
   - Other columns are ignored
3. In `configs/config.yaml`, set:
   ```yaml
   kenpom:
     local_path: data/raw/kenpom/kenpom_2026.csv
   ```

## 6. (Optional) Update Bracket After First Four

First Four games play March 17–18. The bracket CSV ships with placeholder names for those 4 slots.
After results are in, update `data/raw/bracket/bracket_2026.csv`:
- **S16**: Prairie View A&M vs Lehigh → update `Prairie View A&M` row with winner's name
- **W11**: Texas vs NC State → update `Texas` row with winner's name
- **M16**: UMBC vs Howard → update `UMBC` row with winner's name
- **M11**: SMU vs Miami (OH) → update `SMU` row with winner's name

## 7. Run the Pipeline

```bash
python pipeline.py --year 2026
```

Or stage by stage:
```bash
python scripts/ingest_data.py
python scripts/build_features.py
python scripts/train_models.py
python scripts/calibrate_models.py
python scripts/generate_matchups.py
python scripts/run_simulations.py
python scripts/build_field_model.py
python scripts/optimize_brackets.py
python scripts/export_bracket.py
```
```

- [ ] **Step 2: Commit**

```bash
git add SETUP.md
git commit -m "docs: add full setup guide for Kaggle, Barttorvik, and KenPom data"
```

---

### Task 4: Create the 2026 bracket CSV (pre-populated, 64 teams)

The pipeline operates on 64 teams. First Four produces the actual 11/16 seeds — the CSV ships with one team per slot as a placeholder (the team that Barttorvik rates higher). The user updates those 4 rows after First Four results are in.

**Files:**
- Create: `data/raw/bracket/bracket_2026.csv`

- [ ] **Step 1: Create directory and file**

```bash
mkdir -p data/raw/bracket
```

Create `data/raw/bracket/bracket_2026.csv`:

```csv
team_name,seed,region,slot
Duke,1,East,E1
Siena,16,East,E16
Ohio St.,8,East,E8
TCU,9,East,E9
St. John's,5,East,E5
N. Iowa,12,East,E12
Kansas,4,East,E4
Cal Baptist,13,East,E13
Louisville,6,East,E6
South Florida,11,East,E11
Michigan St.,3,East,E3
N. Dakota St.,14,East,E14
UCLA,7,East,E7
UCF,10,East,E10
UConn,2,East,E2
Furman,15,East,E15
Florida,1,South,S1
Prairie View A&M,16,South,S16
Clemson,8,South,S8
Iowa,9,South,S9
Vanderbilt,5,South,S5
McNeese,12,South,S12
Nebraska,4,South,S4
Troy,13,South,S13
North Carolina,6,South,S6
VCU,11,South,S11
Illinois,3,South,S3
Penn,14,South,S14
Saint Mary's,7,South,S7
Texas A&M,10,South,S10
Houston,2,South,S2
Idaho,15,South,S15
Arizona,1,West,W1
LIU,16,West,W16
Villanova,8,West,W8
Utah St.,9,West,W9
Wisconsin,5,West,W5
High Point,12,West,W12
Arkansas,4,West,W4
Hawaii,13,West,W13
BYU,6,West,W6
Texas,11,West,W11
Gonzaga,3,West,W3
Kennesaw St.,14,West,W14
Miami,7,West,W7
Missouri,10,West,W10
Purdue,2,West,W2
Queens,15,West,W15
Michigan,1,Midwest,M1
UMBC,16,Midwest,M16
Georgia,8,Midwest,M8
Saint Louis,9,Midwest,M9
Texas Tech,5,Midwest,M5
Akron,12,Midwest,M12
Alabama,4,Midwest,M4
Hofstra,13,Midwest,M13
Tennessee,6,Midwest,M6
SMU,11,Midwest,M11
Virginia,3,Midwest,M3
Wright St.,14,Midwest,M14
Kentucky,7,Midwest,M7
Santa Clara,10,Midwest,M10
Iowa St.,2,Midwest,M2
Tennessee St.,15,Midwest,M15
```

> **First Four placeholders:** S16=Prairie View A&M, W11=Texas, M16=UMBC, M11=SMU. Update these rows with actual winners after March 17–18.

- [ ] **Step 2: Commit**

```bash
git add data/raw/bracket/bracket_2026.csv
git commit -m "data: add pre-populated 2026 NCAA bracket CSV (64 teams, First Four TBD)"
```

---

### Task 5: Create stat CSV templates and add ingest tests

Templates ship with headers only. They are not pointed at by config until the user has pasted data (see Task 2 and SETUP.md). This section also adds the missing local-CSV ingest tests for Barttorvik and KenPom.

**Files:**
- Create: `data/raw/barttorvik/barttorvik_2026.csv`
- Create: `data/raw/kenpom/kenpom_2026.csv`
- Modify: `tests/test_ingest_connectors.py`

- [ ] **Step 1: Create CSV templates**

```bash
mkdir -p data/raw/barttorvik data/raw/kenpom
```

Create `data/raw/barttorvik/barttorvik_2026.csv`:
```csv
team_name,adj_o,adj_d,tempo
```

Create `data/raw/kenpom/kenpom_2026.csv`:
```csv
team_name,adj_em,adj_t,luck
```

- [ ] **Step 2: Write failing tests for Barttorvik and KenPom local CSV ingest**

Append to `tests/test_ingest_connectors.py`:

```python
from src.data.ingest_barttorvik import ingest_barttorvik
from src.data.ingest_kenpom import ingest_kenpom


def test_ingest_barttorvik_from_local_csv(tmp_path: Path):
    """Barttorvik ingest reads local CSV and normalizes to expected columns."""
    bt_csv = tmp_path / "barttorvik.csv"
    pd.DataFrame([
        {"team_name": "Duke", "adj_o": 120.5, "adj_d": 88.3, "tempo": 71.2},
        {"team_name": "Kansas", "adj_o": 118.1, "adj_d": 91.0, "tempo": 68.5},
    ]).to_csv(bt_csv, index=False)

    cfg = {
        "project": {"target_year": 2026},
        "data": {"random_seed": 42, "num_teams": 64},
        "data_sources": {"barttorvik": {"local_path": str(bt_csv)}},
        "simulation": {"num_sims": 10},
        "optimization": {"min_champion_prob": 0.02, "risk_tolerance": "balanced", "pool_size": 50},
    }
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, cfg)

    out_path = ingest_barttorvik(2026, str(cfg_path))
    out_df = pd.read_csv(out_path)

    assert len(out_df) == 2
    assert "team_name" in out_df.columns
    assert "adj_o" in out_df.columns
    assert "adj_d" in out_df.columns
    assert "tempo" in out_df.columns
    assert out_df.loc[out_df["team_name"] == "Duke", "adj_o"].iloc[0] == pytest.approx(120.5)


def test_ingest_kenpom_from_local_csv(tmp_path: Path):
    """KenPom ingest reads local CSV with required columns and normalizes correctly."""
    kp_csv = tmp_path / "kenpom.csv"
    pd.DataFrame([
        {"team_name": "Duke", "adj_em": 32.1, "adj_t": 70.5, "luck": 0.021},
        {"team_name": "Kansas", "adj_em": 28.4, "adj_t": 68.2, "luck": -0.011},
    ]).to_csv(kp_csv, index=False)

    cfg = {
        "project": {"target_year": 2026},
        "data": {"random_seed": 42, "num_teams": 64},
        "data_sources": {"kenpom": {"local_path": str(kp_csv)}},
        "simulation": {"num_sims": 10},
        "optimization": {"min_champion_prob": 0.02, "risk_tolerance": "balanced", "pool_size": 50},
    }
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, cfg)

    out_path = ingest_kenpom(2026, str(cfg_path))
    out_df = pd.read_csv(out_path)

    assert len(out_df) == 2
    assert set(out_df.columns) == {"season", "team_name", "adj_em", "adj_t", "luck"}
    assert out_df.loc[out_df["team_name"] == "Duke", "adj_em"].iloc[0] == pytest.approx(32.1)
```

Also add `import pytest` at the top of the test file if not already present.

- [ ] **Step 3: Run tests to confirm they fail**

```bash
pytest tests/test_ingest_connectors.py::test_ingest_barttorvik_from_local_csv tests/test_ingest_connectors.py::test_ingest_kenpom_from_local_csv -v
```
Expected: `ImportError` — `ingest_barttorvik` and `ingest_kenpom` are not yet imported in the test file.

- [ ] **Step 4: Read `src/data/ingest_barttorvik.py` to confirm expected output columns**

```bash
python -c "
import pandas as pd, yaml
from src.data.ingest_barttorvik import ingest_barttorvik
"
```
Expected: no error (confirms import works from project root).

- [ ] **Step 5: Run tests to confirm they pass**

The import fix in Step 3 (adding the import lines to the test file) should be sufficient for the tests to find the functions. Run:

```bash
pytest tests/test_ingest_connectors.py -v
```
Expected: all tests PASS including the 2 new ones.

- [ ] **Step 6: Commit**

```bash
git add data/raw/barttorvik/barttorvik_2026.csv data/raw/kenpom/kenpom_2026.csv tests/test_ingest_connectors.py
git commit -m "data: add stat CSV templates; test: add barttorvik and kenpom local-CSV ingest tests"
```

---

## Chunk 2: Kaggle download script

### Task 6: Write tests for Kaggle processing logic

The Kaggle API download requires credentials and network access, so we test only the pure data-processing functions with local fixture data.

**Files:**
- Create: `tests/test_download_data.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_download_data.py`:

```python
from pathlib import Path
import pandas as pd
import pytest
from scripts.download_data import process_game_results, process_teams


def test_process_game_results_filters_by_single_season(tmp_path):
    """Only rows matching the target season are kept."""
    csv = tmp_path / "results.csv"
    pd.DataFrame([
        {"Season": 2025, "WTeamID": 1101, "LTeamID": 1102, "WScore": 70, "LScore": 60, "DayNum": 10},
        {"Season": 2026, "WTeamID": 1103, "LTeamID": 1104, "WScore": 75, "LScore": 68, "DayNum": 12},
        {"Season": 2026, "WTeamID": 1104, "LTeamID": 1103, "WScore": 80, "LScore": 71, "DayNum": 14},
    ]).to_csv(csv, index=False)

    result = process_game_results(csv, season=2026)

    assert len(result) == 2
    assert all(result["Season"] == 2026)


def test_process_game_results_filters_by_multiple_seasons(tmp_path):
    """Passing a list of seasons keeps all matching rows."""
    csv = tmp_path / "results.csv"
    pd.DataFrame([
        {"Season": 2024, "WTeamID": 1, "LTeamID": 2, "WScore": 70, "LScore": 60, "DayNum": 1},
        {"Season": 2025, "WTeamID": 3, "LTeamID": 4, "WScore": 72, "LScore": 65, "DayNum": 2},
        {"Season": 2026, "WTeamID": 5, "LTeamID": 6, "WScore": 68, "LScore": 62, "DayNum": 3},
        {"Season": 2027, "WTeamID": 7, "LTeamID": 8, "WScore": 81, "LScore": 74, "DayNum": 4},
    ]).to_csv(csv, index=False)

    result = process_game_results(csv, season=[2024, 2025, 2026])

    assert len(result) == 3
    assert set(result["Season"].unique()) == {2024, 2025, 2026}


def test_process_teams_returns_required_columns(tmp_path):
    """process_teams returns a DataFrame with TeamID and TeamName columns."""
    csv = tmp_path / "MTeams.csv"
    pd.DataFrame([
        {"TeamID": 1181, "TeamName": "Duke", "FirstD1Season": 1985, "LastD1Season": 2026},
        {"TeamID": 1242, "TeamName": "Kansas", "FirstD1Season": 1985, "LastD1Season": 2026},
    ]).to_csv(csv, index=False)

    result = process_teams(csv)

    assert "TeamID" in result.columns
    assert "TeamName" in result.columns
    assert len(result) == 2
    # Extra columns should not be present
    assert list(result.columns) == ["TeamID", "TeamName"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_download_data.py -v
```
Expected: `ImportError` — `scripts/download_data.py` doesn't exist yet.

---

### Task 7: Implement `scripts/download_data.py`

**Files:**
- Create: `scripts/download_data.py`

- [ ] **Step 1: Implement the script**

Create `scripts/download_data.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/test_download_data.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/download_data.py tests/test_download_data.py
git commit -m "feat: add Kaggle download script with season filtering and team lookup"
```

---

## Chunk 3: Fix crosswalk to use real Kaggle team IDs

### Task 8: Write tests for the updated crosswalk

**Background:** `build_crosswalk` currently hardcodes its barttorvik path as `root / "data" / "raw" / "barttorvik" / f"barttorvik_{year}.csv"` where `root` is always the real project root (set in `src/common/config.py:9`). Task 9 changes this to read the path from `cfg["data_sources"]["barttorvik"]["local_path"]` (falling back to the constructed path if null), making it consistent with other ingest functions and testable via `tmp_path`.

**Files:**
- Modify: `tests/test_ingest_connectors.py`

- [ ] **Step 1: Append crosswalk tests**

Append to `tests/test_ingest_connectors.py`:

```python
from src.data.build_crosswalk import build_crosswalk


def test_build_crosswalk_uses_real_kaggle_team_ids(tmp_path: Path):
    """When MTeams.csv is present, crosswalk assigns real Kaggle TeamIDs."""
    bt_csv = tmp_path / "barttorvik_2026.csv"
    pd.DataFrame([
        {"team_name": "Duke", "adj_o": 120.0, "adj_d": 90.0, "tempo": 70.0},
        {"team_name": "Kansas", "adj_o": 118.0, "adj_d": 92.0, "tempo": 68.0},
    ]).to_csv(bt_csv, index=False)

    mteams_path = tmp_path / "MTeams.csv"
    pd.DataFrame([
        {"TeamID": 1181, "TeamName": "Duke"},
        {"TeamID": 1242, "TeamName": "Kansas"},
        {"TeamID": 1999, "TeamName": "Some Other Team"},
    ]).to_csv(mteams_path, index=False)

    cfg = {
        "project": {"target_year": 2026},
        "data": {"random_seed": 42, "num_teams": 64},
        "data_sources": {
            "barttorvik": {"local_path": str(bt_csv)},
            "kenpom": {"local_path": None},
            "kaggle": {"local_path": None},
            "bracket": {"local_path": None},
        },
        "crosswalk": {"mteams_path": str(mteams_path)},
        "simulation": {"num_sims": 10},
        "optimization": {"min_champion_prob": 0.02, "risk_tolerance": "balanced", "pool_size": 50},
    }
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, cfg)

    out_path = build_crosswalk(2026, str(cfg_path))
    cw = pd.read_csv(out_path)

    duke_id = cw.loc[cw["barttorvik_name"] == "Duke", "kaggle_team_id"].iloc[0]
    kansas_id = cw.loc[cw["barttorvik_name"] == "Kansas", "kaggle_team_id"].iloc[0]
    assert duke_id == 1181
    assert kansas_id == 1242


def test_build_crosswalk_falls_back_to_sequential_without_mteams(tmp_path: Path):
    """When no mteams_path is configured, crosswalk falls back to sequential IDs."""
    bt_csv = tmp_path / "barttorvik_2026.csv"
    pd.DataFrame([
        {"team_name": "Duke", "adj_o": 120.0, "adj_d": 90.0, "tempo": 70.0},
    ]).to_csv(bt_csv, index=False)

    cfg = {
        "project": {"target_year": 2026},
        "data": {"random_seed": 42, "num_teams": 64},
        "data_sources": {
            "barttorvik": {"local_path": str(bt_csv)},
            "kenpom": {"local_path": None},
            "kaggle": {"local_path": None},
            "bracket": {"local_path": None},
        },
        "simulation": {"num_sims": 10},
        "optimization": {"min_champion_prob": 0.02, "risk_tolerance": "balanced", "pool_size": 50},
    }
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, cfg)

    out_path = build_crosswalk(2026, str(cfg_path))
    cw = pd.read_csv(out_path)

    assert len(cw) == 1
    assert cw.iloc[0]["kaggle_team_id"] == 1  # sequential fallback
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_ingest_connectors.py::test_build_crosswalk_uses_real_kaggle_team_ids -v
```
Expected: FAIL — crosswalk currently ignores config paths and uses sequential IDs.

---

### Task 9: Fix `src/data/build_crosswalk.py`

**Files:**
- Modify: `src/data/build_crosswalk.py`

- [ ] **Step 1: Implement fuzzy name matching**

Replace `src/data/build_crosswalk.py` entirely with:

```python
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
```

- [ ] **Step 2: Run all crosswalk tests**

```bash
pytest tests/test_ingest_connectors.py -v
```
Expected: all tests PASS.

- [ ] **Step 3: Run full test suite**

```bash
pytest -v
```
Expected: all tests PASS, no regressions.

- [ ] **Step 4: Commit**

```bash
git add src/data/build_crosswalk.py tests/test_ingest_connectors.py
git commit -m "fix: crosswalk uses real Kaggle TeamIDs via fuzzy name matching, falls back to sequential"
```

---

## Final Verification

- [ ] **Confirm all new files exist**

```bash
ls data/raw/bracket/bracket_2026.csv
ls data/raw/barttorvik/barttorvik_2026.csv
ls data/raw/kenpom/kenpom_2026.csv
ls SETUP.md
ls scripts/download_data.py
```

- [ ] **Confirm config is correct**

```bash
python -c "
from src.common.config import load_config
cfg = load_config('configs/config.yaml')
ds = cfg['data_sources']
print('bracket:', ds['bracket'])
print('kaggle:', ds['kaggle'])
print('barttorvik:', ds['barttorvik'])
print('kenpom:', ds['kenpom'])
print('num_teams:', cfg['data']['num_teams'])
"
```
Expected: bracket and kaggle have `local_path` set; barttorvik and kenpom have `local_path: null`.

- [ ] **Run full test suite one final time**

```bash
pytest -v
```
Expected: all green.

---

## What the user does after implementation

1. `python scripts/download_data.py` — download Kaggle data
2. Paste Barttorvik Team Stats into `data/raw/barttorvik/barttorvik_2026.csv`
3. Paste KenPom stats into `data/raw/kenpom/kenpom_2026.csv`
4. In `configs/config.yaml`, set `local_path` for barttorvik and kenpom
5. (After March 17–18) Update First Four slots in bracket CSV
6. `python pipeline.py --year 2026`
