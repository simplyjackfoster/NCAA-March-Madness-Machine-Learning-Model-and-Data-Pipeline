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
