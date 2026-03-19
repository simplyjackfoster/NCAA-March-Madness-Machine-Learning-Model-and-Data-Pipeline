# Upset Prediction Improvement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the model predicting only 8 upsets (vs. historical avg 17.6) by replacing synthetic training labels with real tournament outcomes and adding Massey ranking differentials as empirically validated features.

**Architecture:** Three coordinated changes: (1) `game_features.py` is rewritten to build training data from historical Kaggle tournament results + Massey ordinals instead of synthetic noise; (2) `team_features.py` and `matchup_matrix.py` are updated to include Massey ranking differentials for the prediction year; (3) `matchup_matrix.py` gets a seed-pair blending layer that anchors probabilities to empirical historical upset rates.

**Tech Stack:** pandas, numpy, scikit-learn, pytest. All source data already present in `data/raw/kaggle/downloads/`.

---

## File Map

| File | Change |
|---|---|
| `src/features/game_features.py` | Rewrite: real Kaggle outcomes + Massey features instead of synthetic labels |
| `src/features/feature_registry.py` | Update: replace old game features with new ones |
| `src/data/build_calibration_set.py` | Update: FEATURES constant to match new feature list |
| `src/models/prior_model.py` | Update: feature column list |
| `src/models/lgbm_model.py` | Update: feature column list |
| `src/models/xgb_model.py` | Update: feature column list |
| `src/models/ensemble.py` | Update: feature_cols list + weight-indexing (elo_diff → seed_diff, net_rating_diff → rank_diff_POM, tempo_diff → rank_diff_MOR) |
| `src/features/team_features.py` | Add: 3 Massey rank columns for prediction year |
| `src/simulation/matchup_matrix.py` | Update: new FEATURES list + seed-pair blending after calibration |
| `tests/test_features.py` | Update: existing game_features tests + add new tests |
| `tests/test_simulation.py` | Update: existing matchup_matrix tests + ensemble test to use new feature list |

---

## Chunk 1: Real Tournament Training Data

### Task 1: Rewrite `game_features.py` to use real tournament outcomes

**Files:**
- Modify: `src/features/game_features.py`
- Modify: `tests/test_features.py`

**Context:** Currently `build_game_features` generates ~1200 synthetic games with `label = int(diff + rng.normal(0, 2) > 0)`. This preserves the rating hierarchy too faithfully — the model never sees a 12-seed beat a 5-seed in training. We replace this with all real NCAA tournament games 2003–2025 from Kaggle, using seed_diff + Massey ranking differentials as features.

The function still writes `data/processed/train.parquet` (consumed by `build_calibration_set.py`) and `data/features/games_{year}.parquet`. The `tourney.parquet` output is removed (nothing reads it).

New FEATURES = `["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]`

- [ ] **Step 1: Write failing test for real-outcome training data**

Add to `tests/test_features.py`:

```python
def test_game_features_uses_real_outcomes(tmp_path):
    """build_game_features must produce binary labels from real tournament results,
    with rank_diff_POM/MOR/SAG columns present, and NO synthetic rng.normal labels."""
    import yaml
    from src.features.game_features import build_game_features

    config = {
        "project": {
            "target_year": 2026,
            "base_data_dir": str(tmp_path / "data"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "outputs_dir": str(tmp_path / "outputs"),
        },
        "data": {"random_seed": 42, "num_teams": 4},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    kaggle_dir = tmp_path / "data" / "raw" / "kaggle" / "downloads"
    kaggle_dir.mkdir(parents=True)

    # 2 historical games
    import pandas as pd
    pd.DataFrame({
        "Season": [2023, 2023],
        "WTeamID": [1101, 1102],
        "LTeamID": [1103, 1104],
    }).to_csv(kaggle_dir / "MNCAATourneyCompactResults.csv", index=False)

    pd.DataFrame({
        "Season": [2023] * 4,
        "TeamID": [1101, 1102, 1103, 1104],
        "Seed": ["W01", "W05", "W12", "W08"],
    }).to_csv(kaggle_dir / "MNCAATourneySeeds.csv", index=False)

    pd.DataFrame({
        "Season": [2023] * 8,
        "RankingDayNum": [128] * 8,
        "SystemName": ["POM"] * 4 + ["MOR"] * 4,
        "TeamID": [1101, 1102, 1103, 1104, 1101, 1102, 1103, 1104],
        "OrdinalRank": [5, 20, 60, 30, 8, 22, 55, 28],
    }).to_csv(kaggle_dir / "MMasseyOrdinals.csv", index=False)

    out = build_game_features(2026, str(cfg_path))
    df = pd.read_parquet(out)

    assert "label" in df.columns
    assert set(df["label"].unique()).issubset({0, 1}), "Labels must be binary"
    assert "rank_diff_POM" in df.columns
    assert "rank_diff_MOR" in df.columns
    assert "rank_diff_SAG" in df.columns, "rank_diff_SAG must be present even if NaN-filled (SAG missing from fixture)"
    assert len(df) == 4, "2 games × 2 mirror rows = 4 rows"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /Users/jackfoster/NCAA-March-Madness-Machine-Learning-Model-and-Data-Pipeline
python -m pytest tests/test_features.py::test_game_features_uses_real_outcomes -v
```

Expected: FAIL (function still uses synthetic labels)

- [ ] **Step 3: Rewrite `src/features/game_features.py`**

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config

FEATURES = ["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]
_MASSEY_SYSTEMS = ["POM", "MOR", "SAG"]
_MASSEY_DAY_CUTOFF = 133  # selection Sunday neighborhood


def _load_seed_map(kaggle_dir: Path) -> pd.DataFrame:
    """Returns DataFrame with columns [Season, TeamID, seed_num]."""
    seeds = pd.read_csv(kaggle_dir / "MNCAATourneySeeds.csv")
    seeds["seed_num"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)
    return seeds[["Season", "TeamID", "seed_num"]]


def _load_massey_ranks(kaggle_dir: Path) -> pd.DataFrame:
    """Returns DataFrame with columns [Season, TeamID, rank_POM, rank_MOR, rank_SAG].
    Uses the latest pre-tournament ranking per system per team per season.
    Missing systems get NaN (handled downstream by fill with median).
    """
    massey = pd.read_csv(kaggle_dir / "MMasseyOrdinals.csv")
    pre = massey[massey["RankingDayNum"] <= _MASSEY_DAY_CUTOFF]
    # Keep latest ranking day per system/team/season
    latest = (
        pre.sort_values("RankingDayNum")
        .groupby(["Season", "SystemName", "TeamID"])
        .tail(1)
    )
    result = None
    for system in _MASSEY_SYSTEMS:
        sys_df = (
            latest[latest["SystemName"] == system][["Season", "TeamID", "OrdinalRank"]]
            .rename(columns={"OrdinalRank": f"rank_{system}"})
        )
        if result is None:
            result = sys_df
        else:
            result = result.merge(sys_df, on=["Season", "TeamID"], how="outer")
    return result if result is not None else pd.DataFrame()


def build_game_features(year: int, config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    root = cfg["_root"]
    kaggle_dir = root / "data" / "raw" / "kaggle" / "downloads"

    results = pd.read_csv(kaggle_dir / "MNCAATourneyCompactResults.csv")
    seed_map = _load_seed_map(kaggle_dir)
    rank_map = _load_massey_ranks(kaggle_dir)

    # Build one row per game from winner's perspective, then mirror
    rows = []
    for _, game in results.iterrows():
        season = int(game["Season"])
        w, l = int(game["WTeamID"]), int(game["LTeamID"])

        def get_seed(tid):
            s = seed_map[(seed_map["Season"] == season) & (seed_map["TeamID"] == tid)]
            return int(s["seed_num"].iloc[0]) if len(s) else 8

        def get_rank(tid, system):
            if rank_map is None or rank_map.empty:
                return np.nan
            col = f"rank_{system}"
            if col not in rank_map.columns:
                return np.nan
            r = rank_map[(rank_map["Season"] == season) & (rank_map["TeamID"] == tid)]
            return float(r[col].iloc[0]) if len(r) else np.nan

        ws, ls = get_seed(w), get_seed(l)
        row_w = {
            "season": season,
            "team_id": w,
            "opp_id": l,
            "seed_diff": ws - ls,
            "label": 1,
        }
        row_l = {
            "season": season,
            "team_id": l,
            "opp_id": w,
            "seed_diff": ls - ws,
            "label": 0,
        }
        for system in _MASSEY_SYSTEMS:
            wr = get_rank(w, system)
            lr = get_rank(l, system)
            row_w[f"rank_diff_{system}"] = wr - lr if not (np.isnan(wr) or np.isnan(lr)) else np.nan
            row_l[f"rank_diff_{system}"] = lr - wr if not (np.isnan(wr) or np.isnan(lr)) else np.nan
        rows.append(row_w)
        rows.append(row_l)

    df = pd.DataFrame(rows)

    # Fill missing Massey ranks with median (teams not covered by a system)
    for col in [f"rank_diff_{s}" for s in _MASSEY_SYSTEMS]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    out_path = root / "data" / "features" / f"games_{year}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / "data" / "processed" / "train.parquet", index=False)
    return out_path
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
python -m pytest tests/test_features.py::test_game_features_uses_real_outcomes -v
```

Expected: PASS

- [ ] **Step 5: Update `build_calibration_set.py` FEATURES constant**

In `src/data/build_calibration_set.py`, change line 12:

```python
# Old:
FEATURES = ["elo_diff", "net_rating_diff", "tempo_diff", "seed_diff"]

# New:
FEATURES = ["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]
```

- [ ] **Step 6: Update existing test that creates train.parquet with old columns**

In `tests/test_features.py`, find `test_build_calibration_set_produces_valid_probs` and replace the synthetic training data block (around line 142–154) with data using the new feature columns:

```python
    rng = np.random.default_rng(42)
    n = 200
    seed_diff = rng.integers(-15, 15, n)
    pd.DataFrame({
        "seed_diff": seed_diff,
        "rank_diff_POM": seed_diff * 4.0 + rng.normal(0, 5, n),
        "rank_diff_MOR": seed_diff * 4.0 + rng.normal(0, 5, n),
        "rank_diff_SAG": seed_diff * 4.0 + rng.normal(0, 5, n),
        "label": (seed_diff + rng.normal(0, 3, n) < 0).astype(int),
    }).to_parquet(proc_dir / "train.parquet", index=False)
```

- [ ] **Step 7: Update existing `test_game_features_includes_seed_diff`**

This test creates a team_season parquet and calls `build_game_features` — but the new implementation reads from Kaggle CSVs, not team_season parquet. Replace it with a check that the new function writes `train.parquet` and `games_{year}.parquet`:

```python
def test_game_features_writes_output_files(tmp_path):
    """build_game_features must write games_{year}.parquet and train.parquet."""
    import yaml
    from src.features.game_features import build_game_features

    config = {
        "project": {
            "target_year": 2026,
            "base_data_dir": str(tmp_path / "data"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "outputs_dir": str(tmp_path / "outputs"),
        },
        "data": {"random_seed": 42, "num_teams": 4},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    import pandas as pd
    kaggle_dir = tmp_path / "data" / "raw" / "kaggle" / "downloads"
    kaggle_dir.mkdir(parents=True)
    pd.DataFrame({"Season": [2023], "WTeamID": [1101], "LTeamID": [1103]}).to_csv(
        kaggle_dir / "MNCAATourneyCompactResults.csv", index=False
    )
    pd.DataFrame({
        "Season": [2023, 2023], "TeamID": [1101, 1103], "Seed": ["W01", "W12"],
    }).to_csv(kaggle_dir / "MNCAATourneySeeds.csv", index=False)
    pd.DataFrame({
        "Season": [2023, 2023], "RankingDayNum": [128, 128],
        "SystemName": ["POM", "POM"], "TeamID": [1101, 1103], "OrdinalRank": [5, 60],
    }).to_csv(kaggle_dir / "MMasseyOrdinals.csv", index=False)

    out = build_game_features(2026, str(cfg_path))
    assert out.exists()
    assert (tmp_path / "data" / "processed" / "train.parquet").exists()
    df = pd.read_parquet(out)
    assert "label" in df.columns
    assert "seed_diff" in df.columns
```

- [ ] **Step 8: Run all feature tests**

```bash
python -m pytest tests/test_features.py -v
```

Expected: all PASS

- [ ] **Step 9: Update `feature_registry.py` to reflect new game features**

In `src/features/feature_registry.py`, replace the three old game-level entries and add the new ones:

```python
# Remove these three lines:
FeatureSpec("elo_diff", "game", "Elo difference team - opponent"),
FeatureSpec("net_rating_diff", "game", "Net rating difference team - opponent"),
FeatureSpec("tempo_diff", "game", "Tempo difference team - opponent"),

# Add these four lines:
FeatureSpec("seed_diff", "game", "Seed difference team - opponent"),
FeatureSpec("rank_diff_POM", "game", "KenPom (POM) Massey ordinal rank difference"),
FeatureSpec("rank_diff_MOR", "game", "Massey (MOR) ordinal rank difference"),
FeatureSpec("rank_diff_SAG", "game", "Sagarin (SAG) ordinal rank difference"),
```

Also update `test_feature_registry_has_core_features` in `tests/test_features.py`:

```python
def test_feature_registry_has_core_features():
    names = get_feature_names()
    assert "seed_diff" in names
    assert "rank_diff_POM" in names
    assert "net_rating" in names  # team-level feature, unchanged
```

- [ ] **Step 10: Run all feature tests**

```bash
python -m pytest tests/test_features.py -v
```

Expected: all PASS

- [ ] **Step 11: Commit**

```bash
git add src/features/game_features.py src/features/feature_registry.py src/data/build_calibration_set.py tests/test_features.py
git commit -m "feat: replace synthetic training labels with real tournament outcomes"
```

---

### Task 2: Update model trainers and ensemble to use new feature list

**Files:**
- Modify: `src/models/prior_model.py`
- Modify: `src/models/lgbm_model.py`
- Modify: `src/models/xgb_model.py`
- Modify: `src/models/ensemble.py`
- Modify: `tests/test_simulation.py`

**Context:** All four model files hard-code `["elo_diff", "net_rating_diff", "tempo_diff", "seed_diff"]`. After Task 1 rewrites `train.parquet`, these files will raise `KeyError` on `elo_diff`. Update all four to use `["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]`. The `ensemble.py` weight-indexing also maps feature names to model names — update those mappings. Two existing tests in `test_simulation.py` also embed the old feature list and need updating.

- [ ] **Step 1: Update `src/models/prior_model.py` line 17**

```python
# Old:
x = train[["elo_diff", "net_rating_diff", "tempo_diff", "seed_diff"]]
# New:
x = train[["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]]
```

- [ ] **Step 2: Update `src/models/lgbm_model.py` line 18**

```python
# Old:
x = train[["elo_diff", "net_rating_diff", "tempo_diff", "seed_diff"]]
# New:
x = train[["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]]
```

- [ ] **Step 3: Update `src/models/xgb_model.py` line 18**

```python
# Old:
x = train[["elo_diff", "net_rating_diff", "tempo_diff", "seed_diff"]]
# New:
x = train[["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]]
```

- [ ] **Step 4: Update `src/models/ensemble.py`**

Update `feature_cols` (line 17) and the three weight-indexing lines (lines 24–26):

```python
# Old:
feature_cols = ["elo_diff", "net_rating_diff", "tempo_diff", "seed_diff"]
...
weights = {
    "prior": float(strength["elo_diff"] / total),
    "lgbm": float(strength["net_rating_diff"] / total),
    "xgb": float(strength["tempo_diff"] / total),
}

# New:
feature_cols = ["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]
...
weights = {
    "prior": float(strength["seed_diff"] / total),
    "lgbm": float(strength["rank_diff_POM"] / total),
    "xgb": float(strength["rank_diff_MOR"] / total),
}
```

- [ ] **Step 5: Update `test_ensemble_weights_constant_feature` in `tests/test_simulation.py`**

The existing test writes a `train.parquet` with old columns including `tempo_diff` (constant). Update it to use new features, keeping one constant column to verify the zero-variance guard still works:

```python
    pd.DataFrame({
        "seed_diff": elo * -0.5,
        "rank_diff_POM": elo,
        "rank_diff_MOR": np.zeros(100),   # constant — will produce NaN in corrwith
        "rank_diff_SAG": elo * 0.8,
        "label": (elo > 0).astype(int),
    }).to_parquet(proc_dir / "train.parquet", index=False)
```

- [ ] **Step 6: Update `test_matchup_matrix_no_feature_name_warning` in `tests/test_simulation.py`**

This test trains a model and builds team features using old columns. Update the `FEATURES` constant, training DataFrame, and team features DataFrame to use new columns. Also add Massey rank columns to the team features parquet and a Kaggle dir for seed-pair blending:

```python
    FEATURES = ["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]
    rng = np.random.default_rng(42)
    n = 100
    seed = rng.integers(-15, 15, n).astype(float)
    X = pd.DataFrame({
        "seed_diff": seed,
        "rank_diff_POM": seed * 4.0 + rng.normal(0, 5, n),
        "rank_diff_MOR": seed * 4.0 + rng.normal(0, 5, n),
        "rank_diff_SAG": seed * 4.0 + rng.normal(0, 5, n),
    })
    y = (seed < 0).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)
```

Add Massey rank columns to the team_season parquet:

```python
    pd.DataFrame({
        "season": [year] * 4, "kaggle_team_id": [1, 2, 3, 4],
        "display_name": ["A", "B", "C", "D"],
        "adj_o": [110.0, 100.0, 90.0, 80.0], "adj_d": [80.0, 90.0, 100.0, 110.0],
        "tempo": [70.0, 68.0, 66.0, 64.0],
        "net_rating": [30.0, 10.0, -10.0, -30.0],
        "elo_pre": [1740.0, 1580.0, 1420.0, 1260.0],
        "adj_em": [20.0, 10.0, -5.0, -20.0], "luck": [0.0] * 4, "seed": [1, 4, 5, 8],
        "massey_rank_POM": [10.0, 30.0, 60.0, 100.0],
        "massey_rank_MOR": [12.0, 28.0, 58.0, 95.0],
        "massey_rank_SAG": [8.0, 32.0, 62.0, 98.0],
    }).to_parquet(feat_dir / f"team_season_{year}.parquet", index=False)
```

Add Kaggle dir with minimal CSV files needed by `_load_seed_pair_win_rates`:

```python
    kaggle_dir = tmp_path / "data" / "raw" / "kaggle" / "downloads"
    kaggle_dir.mkdir(parents=True)
    pd.DataFrame({"Season": [2020], "WTeamID": [1], "LTeamID": [4]}).to_csv(
        kaggle_dir / "MNCAATourneyCompactResults.csv", index=False
    )
    pd.DataFrame({"Season": [2020, 2020], "TeamID": [1, 4], "Seed": ["W01", "W08"]}).to_csv(
        kaggle_dir / "MNCAATourneySeeds.csv", index=False
    )
```

- [ ] **Step 7: Update `test_matchup_matrix_probabilities_not_extreme` in `tests/test_simulation.py`**

Apply the same fixture updates as Step 6 — new FEATURES training DataFrame, Massey rank columns in team_season parquet, and Kaggle CSV files. The full replacement for the fixture section (lines 152–175 of test_simulation.py):

```python
    FEATURES = ["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]
    rng = np.random.default_rng(42)
    n = 200
    seed = rng.integers(-15, 15, n).astype(float)
    X = pd.DataFrame({
        "seed_diff": seed,
        "rank_diff_POM": seed * 4.0 + rng.normal(0, 5, n),
        "rank_diff_MOR": seed * 4.0 + rng.normal(0, 5, n),
        "rank_diff_SAG": seed * 4.0 + rng.normal(0, 5, n),
    })
    y = (seed + rng.normal(0, 3, n) < 0).astype(int)
    prior = LogisticRegression(max_iter=1000).fit(X, y)
```

Team features parquet (replace lines 137–149):

```python
    pd.DataFrame({
        "season": [year] * 4,
        "kaggle_team_id": [1, 2, 3, 4],
        "display_name": ["TeamA", "TeamB", "TeamC", "TeamD"],
        "adj_o": [120.0, 100.0, 85.0, 70.0], "adj_d": [70.0, 90.0, 105.0, 120.0],
        "tempo": [70.0, 68.0, 66.0, 64.0],
        "net_rating": [50.0, 10.0, -20.0, -50.0], "elo_pre": [1900.0, 1580.0, 1340.0, 1100.0],
        "adj_em": [30.0, 10.0, -5.0, -20.0], "luck": [0.0] * 4, "seed": [1, 4, 5, 8],
        "massey_rank_POM": [5.0, 25.0, 60.0, 120.0],
        "massey_rank_MOR": [7.0, 28.0, 62.0, 115.0],
        "massey_rank_SAG": [4.0, 22.0, 58.0, 118.0],
    }).to_parquet(feat_dir / f"team_season_{year}.parquet", index=False)
```

Add Kaggle dir (add after the calibrator save block):

```python
    kaggle_dir = tmp_path / "data" / "raw" / "kaggle" / "downloads"
    kaggle_dir.mkdir(parents=True)
    pd.DataFrame({"Season": [2020], "WTeamID": [1], "LTeamID": [4]}).to_csv(
        kaggle_dir / "MNCAATourneyCompactResults.csv", index=False
    )
    pd.DataFrame({"Season": [2020, 2020], "TeamID": [1, 4], "Seed": ["W01", "W08"]}).to_csv(
        kaggle_dir / "MNCAATourneySeeds.csv", index=False
    )
```

- [ ] **Step 8: Also update `loyo_validator.py` FEATURES list**

`src/models/loyo_validator.py` line 14 hard-codes the old feature list and will raise `KeyError` after `train.parquet` changes:

```python
# Old:
FEATURES = ["elo_diff", "net_rating_diff", "tempo_diff"]

# New:
FEATURES = ["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]
```

- [ ] **Step 9: Run ensemble and non-matchup simulation tests only**

Run only the tests that do NOT require the new `matchup_matrix.py` (those come in Chunk 2). The matchup_matrix tests will fail until the rewrite in the next chunk — skip them here.

```bash
python -m pytest tests/test_simulation.py::test_ensemble_weights_constant_feature tests/test_simulation.py::test_xgb_no_use_label_encoder_warning -v
```

Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add src/models/prior_model.py src/models/lgbm_model.py src/models/xgb_model.py src/models/ensemble.py src/models/loyo_validator.py tests/test_simulation.py
git commit -m "feat: update model trainers and loyo_validator to use new Massey-based feature set"
```

---

## Chunk 2: Massey Rankings in Team Features and Matchup Matrix

### Task 3: Add Massey rank columns to `team_features.py`

**Files:**
- Modify: `src/features/team_features.py`
- Modify: `tests/test_features.py`

**Context:** For the prediction year (e.g. 2026), `matchup_matrix.py` needs `massey_rank_POM`, `massey_rank_MOR`, `massey_rank_SAG` per team to compute rank_diff features. These come from `MMasseyOrdinals.csv` using the same latest-pre-tourney logic as game_features.py.

- [ ] **Step 1: Write failing test**

Add to `tests/test_features.py`:

```python
def test_team_features_includes_massey_ranks(tmp_path, monkeypatch):
    """team_season parquet must include massey_rank_POM, massey_rank_MOR, massey_rank_SAG."""
    import yaml
    import pandas as pd
    from src.common import config as config_module

    config = {
        "project": {
            "target_year": 2026,
            "base_data_dir": str(tmp_path / "data"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "outputs_dir": str(tmp_path / "outputs"),
        },
        "data": {"random_seed": 42, "num_teams": 4},
        "data_sources": {},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    def mock_load_config(config_path="configs/config.yaml"):
        import yaml as _yaml
        cfg = _yaml.safe_load(cfg_path.open())
        cfg["_root"] = tmp_path
        return cfg

    monkeypatch.setattr(config_module, "load_config", mock_load_config)

    year = 2026
    bt_dir = tmp_path / "data" / "raw" / "barttorvik"
    bt_dir.mkdir(parents=True)
    pd.DataFrame({
        "season": [year] * 2,
        "team_name": ["TeamA", "TeamB"],
        "adj_o": [110.0, 100.0],
        "adj_d": [90.0, 100.0],
        "tempo": [70.0, 65.0],
    }).to_csv(bt_dir / f"barttorvik_{year}.csv", index=False)

    cx_dir = tmp_path / "data" / "crosswalks"
    cx_dir.mkdir(parents=True)
    pd.DataFrame({
        "barttorvik_name": ["TeamA", "TeamB"],
        "kaggle_team_id": [1101, 1102],
        "display_name": ["TeamA", "TeamB"],
    }).to_csv(cx_dir / "team_id_map.csv", index=False)

    kaggle_dir = tmp_path / "data" / "raw" / "kaggle" / "downloads"
    kaggle_dir.mkdir(parents=True)
    pd.DataFrame({
        "Season": [year, year],
        "RankingDayNum": [128, 128],
        "SystemName": ["POM", "POM"],
        "TeamID": [1101, 1102],
        "OrdinalRank": [15, 45],
    }).to_csv(kaggle_dir / "MMasseyOrdinals.csv", index=False)

    from src.features.team_features import build_team_features
    out = build_team_features(year, "dummy_config")
    df = pd.read_parquet(out)

    assert "massey_rank_POM" in df.columns, "massey_rank_POM missing"
    assert "massey_rank_MOR" in df.columns, "massey_rank_MOR missing (should be NaN-filled)"
    row_a = df[df["display_name"] == "TeamA"].iloc[0]
    assert row_a["massey_rank_POM"] == 15.0
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest tests/test_features.py::test_team_features_includes_massey_ranks -v
```

Expected: FAIL

- [ ] **Step 3: Update `src/features/team_features.py`**

After the KenPom merge block (around line 35), add a Massey ordinals block. Also add the 3 new columns to `out_cols`.

Insert after `teams["luck"] = teams["luck"].fillna(0.0)`:

```python
    # Massey ordinals (POM, MOR, SAG) for this year
    _MASSEY_SYSTEMS = ["POM", "MOR", "SAG"]
    massey_path = root / "data" / "raw" / "kaggle" / "downloads" / "MMasseyOrdinals.csv"
    if massey_path.exists():
        massey = pd.read_csv(massey_path)
        pre = massey[(massey["Season"] == year) & (massey["RankingDayNum"] <= 133)]
        for system in _MASSEY_SYSTEMS:
            sys_df = (
                pre[pre["SystemName"] == system]
                .sort_values("RankingDayNum")
                .groupby("TeamID")
                .tail(1)[["TeamID", "OrdinalRank"]]
                .rename(columns={"OrdinalRank": f"massey_rank_{system}"})
            )
            teams = teams.merge(
                sys_df.rename(columns={"TeamID": "kaggle_team_id"}),
                on="kaggle_team_id",
                how="left",
            )
    for system in ["POM", "MOR", "SAG"]:
        col = f"massey_rank_{system}"
        if col not in teams.columns:
            teams[col] = np.nan
        # Fill missing with median rank (neutral fallback)
        teams[col] = teams[col].fillna(teams[col].median())
```

Also add `import numpy as np` at the top of the file, and add the 3 new columns to `out_cols`:

```python
    out_cols = [
        "season",
        "kaggle_team_id",
        "display_name",
        "adj_o",
        "adj_d",
        "tempo",
        "net_rating",
        "elo_pre",
        "adj_em",
        "luck",
        "seed",
        "massey_rank_POM",
        "massey_rank_MOR",
        "massey_rank_SAG",
    ]
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
python -m pytest tests/test_features.py::test_team_features_includes_massey_ranks -v
```

Expected: PASS

- [ ] **Step 5: Run all feature tests**

```bash
python -m pytest tests/test_features.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/features/team_features.py tests/test_features.py
git commit -m "feat: add Massey ranking columns to team features"
```

---

### Task 4: Update matchup matrix to use new features + seed-pair blending

**Files:**
- Modify: `src/simulation/matchup_matrix.py`
- Modify: `tests/test_simulation.py`

**Context:** `matchup_matrix.py` currently uses `["elo_diff", "net_rating_diff", "tempo_diff", "seed_diff"]`. We update it to `["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]` and add seed-pair blending after Platt calibration.

Blending formula: `final_prob = 0.7 × platt_prob + 0.3 × historical_seed_win_rate`

Historical seed win rates are computed once at the top of `build_matchup_matrix` from `MNCAATourneyCompactResults.csv` + `MNCAATourneySeeds.csv`. For any seed pair not found in history, fall back to the Platt probability unchanged.

- [ ] **Step 1: Read existing simulation tests**

```bash
cat tests/test_simulation.py
```

- [ ] **Step 2: Write failing test for seed-pair blending**

Add to `tests/test_simulation.py` (or create if it doesn't cover matchup_matrix):

```python
def test_matchup_matrix_blends_seed_pair_probabilities(tmp_path, monkeypatch):
    """After build_matchup_matrix, the blending formula must be applied.
    Set historical rate to exactly 0.95 for the 1v16 matchup.
    Assert final_prob is close to 0.7 * platt_prob + 0.3 * 0.95."""
    import yaml, pickle
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from src.simulation.matchup_matrix import build_matchup_matrix
    from src.common import config as config_module

    config = {
        "project": {
            "target_year": 2026,
            "base_data_dir": str(tmp_path / "data"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "outputs_dir": str(tmp_path / "outputs"),
        },
        "data": {"random_seed": 42, "num_teams": 4},
        "data_sources": {},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    def mock_load_config(cp="configs/config.yaml"):
        import yaml as _y
        c = _y.safe_load(cfg_path.open())
        c["_root"] = tmp_path
        return c

    monkeypatch.setattr(config_module, "load_config", mock_load_config)

    # Write team features: 2 teams — seed 1 vs seed 16
    feat_dir = tmp_path / "data" / "features"
    feat_dir.mkdir(parents=True)
    pd.DataFrame({
        "season": [2026, 2026],
        "kaggle_team_id": [1101, 1116],
        "display_name": ["TopSeed", "BottomSeed"],
        "adj_o": [120.0, 95.0], "adj_d": [85.0, 110.0],
        "tempo": [70.0, 65.0], "net_rating": [35.0, -15.0],
        "elo_pre": [1780.0, 1380.0], "adj_em": [35.0, -15.0],
        "luck": [0.0, 0.0], "seed": [1, 16],
        "massey_rank_POM": [5.0, 180.0],
        "massey_rank_MOR": [6.0, 175.0],
        "massey_rank_SAG": [4.0, 185.0],
    }).to_parquet(feat_dir / "team_season_2026.parquet", index=False)

    # Write a dummy trained model
    art_dir = tmp_path / "artifacts" / "models"
    art_dir.mkdir(parents=True)
    lr = LogisticRegression()
    lr.fit([[0, 0, 0, 0], [10, 10, 10, 10]], [0, 1])
    with (art_dir / "prior_model.pkl").open("wb") as f:
        pickle.dump(lr, f)

    # Write a fixed calibrator that always returns 0.5 — pinning Platt output so we
    # can verify the blending formula numerically.
    # Expected for 1v16: 0.7 * 0.5 + 0.3 * (136/138) = 0.350 + 0.296 = 0.646
    class FixedCalibrator:
        def predict_proba(self, X):
            return np.column_stack([
                np.full(len(X), 0.5),
                np.full(len(X), 0.5),
            ])

    cal_dir = tmp_path / "artifacts" / "calibrators"
    cal_dir.mkdir(parents=True)
    with (cal_dir / "isotonic.pkl").open("wb") as f:
        pickle.dump(FixedCalibrator(), f)

    # Write Kaggle files for seed-pair win rates
    kaggle_dir = tmp_path / "data" / "raw" / "kaggle" / "downloads"
    kaggle_dir.mkdir(parents=True)
    # Historical: seed 1 beats seed 16 in 136/138 games historically
    w_ids = [1101] * 136 + [1116] * 2
    l_ids = [1116] * 136 + [1101] * 2
    pd.DataFrame({"Season": [2020] * 138, "WTeamID": w_ids, "LTeamID": l_ids}).to_csv(
        kaggle_dir / "MNCAATourneyCompactResults.csv", index=False
    )
    seed_rows = []
    for tid, seed in [(1101, "W01"), (1116, "W16")]:
        seed_rows.append({"Season": 2020, "TeamID": tid, "Seed": seed})
    pd.DataFrame(seed_rows).to_csv(kaggle_dir / "MNCAATourneySeeds.csv", index=False)

    out = build_matchup_matrix(2026, str(cfg_path))
    mat = np.load(out)

    # FixedCalibrator always returns Platt=0.5.
    # Historical rate for 1v16 = 136/138 ≈ 0.9855.
    # Expected blended value: 0.7 * 0.5 + 0.3 * 0.9855 = 0.6457
    prob_1_beats_16 = mat[0, 1]
    expected = 0.7 * 0.5 + 0.3 * (136 / 138)
    assert abs(prob_1_beats_16 - expected) < 0.05, (
        f"1v16 blended prob {prob_1_beats_16:.4f} expected ~{expected:.4f} — "
        f"blending formula not applied correctly"
    )
```

- [ ] **Step 3: Run test to confirm it fails**

```bash
python -m pytest tests/test_simulation.py::test_matchup_matrix_blends_seed_pair_probabilities -v
```

Expected: FAIL

- [ ] **Step 4: Rewrite `src/simulation/matchup_matrix.py`**

```python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config

FEATURES = ["seed_diff", "rank_diff_POM", "rank_diff_MOR", "rank_diff_SAG"]
_BLEND_WEIGHT = 0.7  # 70% model, 30% historical seed-pair rate


def _load_seed_pair_win_rates(kaggle_dir: Path) -> dict[tuple[int, int], float]:
    """Compute empirical win rate for (winner_seed, loser_seed) pairs from history.
    Returns {(seed_a, seed_b): P(seed_a beats seed_b)} for all observed pairs.
    """
    results = pd.read_csv(kaggle_dir / "MNCAATourneyCompactResults.csv")
    seeds_df = pd.read_csv(kaggle_dir / "MNCAATourneySeeds.csv")
    seeds_df["seed_num"] = seeds_df["Seed"].str.extract(r"(\d+)").astype(int)
    seed_map = dict(zip(
        zip(seeds_df["Season"], seeds_df["TeamID"]),
        seeds_df["seed_num"]
    ))

    counts: dict[tuple[int, int], list[int]] = {}
    for _, row in results.iterrows():
        ws = seed_map.get((row["Season"], row["WTeamID"]))
        ls = seed_map.get((row["Season"], row["LTeamID"]))
        if ws is None or ls is None:
            continue
        pair = (int(ws), int(ls))
        mirror = (int(ls), int(ws))
        counts.setdefault(pair, []).append(1)
        counts.setdefault(mirror, []).append(0)

    return {pair: sum(wins) / len(wins) for pair, wins in counts.items() if wins}


def build_matchup_matrix(year: int, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["_root"]
    team_df = pd.read_parquet(root / "data" / "features" / f"team_season_{year}.parquet")

    with (root / "artifacts" / "models" / "prior_model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (root / "artifacts" / "calibrators" / "isotonic.pkl").open("rb") as f:
        calibrator = pickle.load(f)

    kaggle_dir = root / "data" / "raw" / "kaggle" / "downloads"
    seed_pair_rates = _load_seed_pair_win_rates(kaggle_dir)

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
                    int(a.get("seed", 8)) - int(b.get("seed", 8)),
                    float(a.get("massey_rank_POM", 150)) - float(b.get("massey_rank_POM", 150)),
                    float(a.get("massey_rank_MOR", 150)) - float(b.get("massey_rank_MOR", 150)),
                    float(a.get("massey_rank_SAG", 150)) - float(b.get("massey_rank_SAG", 150)),
                ]],
                columns=FEATURES,
            )
            raw_prob = float(model.predict_proba(x)[0, 1])
            cal_input = np.array([[raw_prob]])
            platt_prob = float(calibrator.predict_proba(cal_input)[0, 1])

            # Seed-pair blending
            seed_a = int(a.get("seed", 8))
            seed_b = int(b.get("seed", 8))
            historical_rate = seed_pair_rates.get((seed_a, seed_b))
            if historical_rate is not None:
                mat[i, j] = _BLEND_WEIGHT * platt_prob + (1 - _BLEND_WEIGHT) * historical_rate
            else:
                mat[i, j] = platt_prob

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
```

- [ ] **Step 5: Run test to confirm it passes**

```bash
python -m pytest tests/test_simulation.py::test_matchup_matrix_blends_seed_pair_probabilities -v
```

Expected: PASS

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all PASS (or only pre-existing failures)

- [ ] **Step 7: Commit**

```bash
git add src/simulation/matchup_matrix.py tests/test_simulation.py
git commit -m "feat: update matchup matrix with Massey features and seed-pair blending"
```

- [ ] **Step 8: Run full simulation test suite (deferred from Task 2)**

Now that `matchup_matrix.py` is rewritten with the new feature set, the matchup_matrix tests can run:

```bash
python -m pytest tests/test_simulation.py -v
```

Expected: all PASS
```

---

## Chunk 3: Integration Validation

### Task 5: Run pipeline and verify upset count

**Files:** No code changes — pipeline run only.

**Context:** Run the full 2026 pipeline end-to-end and confirm predicted upsets fall in the [11, 23] historical range.

- [ ] **Step 1: Run the pipeline**

```bash
cd /Users/jackfoster/NCAA-March-Madness-Machine-Learning-Model-and-Data-Pipeline
python pipeline.py --year 2026
```

Expected: completes without error

- [ ] **Step 2: Count upsets in the bracket output**

```bash
python - << 'EOF'
import numpy as np
import pandas as pd
from pathlib import Path

root = Path(".")
mat = np.load(root / "data" / "tournament" / "prob_matrix_2026.npy")
bracket = pd.read_csv(root / "data" / "raw" / "bracket" / "bracket_2026.csv")
teams = pd.read_parquet(root / "data" / "features" / "team_season_2026.parquet")

merged = bracket.merge(
    teams.reset_index().rename(columns={"index": "team_idx"}),
    left_on="team_name", right_on="display_name", how="left",
)
seed_col = "seed_x" if "seed_x" in merged.columns else "seed"
ordered = merged.sort_values(["slot", seed_col]).dropna(subset=["team_idx"])
alive = ordered["team_idx"].astype(int).tolist()
seeds = dict(zip(ordered["team_idx"].astype(int), ordered[seed_col].astype(int)))

upsets = 0
total = 0
while len(alive) > 1:
    nxt = []
    for i in range(0, len(alive), 2):
        a, b = alive[i], alive[i+1]
        winner = a if mat[a, b] >= 0.5 else b
        loser = b if winner == a else a
        if seeds[winner] > seeds[loser]:
            upsets += 1
        total += 1
        nxt.append(winner)
    alive = nxt

print(f"Predicted upsets: {upsets} / {total} games ({upsets/total:.1%})")
print(f"Historical range: 11-23 (avg 17.6)")
print(f"Status: {'PASS' if 11 <= upsets <= 23 else 'FAIL — outside historical range'}")
EOF
```

Expected: upset count between 11 and 23

- [ ] **Step 3: Verify no extreme probabilities**

```bash
python - << 'EOF'
import numpy as np
mat = np.load("data/tournament/prob_matrix_2026.npy")
max_prob = mat.max()
print(f"Max matchup probability: {max_prob:.3f}")
print(f"Status: {'PASS' if max_prob < 0.98 else 'FAIL — probabilities still too extreme'}")
EOF
```

Expected: max < 0.98

- [ ] **Step 4: Print full bracket to verify output still works**

```bash
python pipeline.py --year 2026
cat outputs/full_bracket_2026.txt | head -60
```

Expected: bracket renders with top-3 picks header and all rounds

- [ ] **Step 5: Commit results**

```bash
git add -u
git commit -m "feat: upset prediction improvement — real outcomes + Massey features + seed-pair blending"
```
