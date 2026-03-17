# March Madness 2026 — Real Data Setup Design

**Date:** 2026-03-16
**Status:** Approved

---

## Goal

Connect the existing fully-implemented pipeline to real 2026 data sources so the user can run the models themselves and generate a bracket recommendation for the 2026 NCAA Tournament.

## Context

The pipeline (`pipeline.py`) is complete across all 6 phases. The only gap is that `configs/config.yaml` has `local_path: null` for all four data sources, causing every stage to fall back to synthetic data. This design wires in real data with minimal friction.

---

## Data Sources

### 1. Bracket — Pre-populated from CBS Sports PDF
- **File:** `data/raw/bracket/bracket_2026.csv`
- **How:** Pre-generated from the CBS Sports 2026 bracket PDF (all 68 teams)
- **Columns:** `team_name, seed, region, slot`
- **Action:** No user action needed — file created automatically

### 2. Barttorvik Team Stats — Manual paste
- **File:** `data/raw/barttorvik/barttorvik_2026.csv`
- **Source:** barttorvik.com → Team Stats page
- **Columns:** `team_name, adj_o, adj_d, tempo`
- **Action:** User copies Team Stats table from Barttorvik and pastes into this CSV

### 3. KenPom — Manual paste
- **File:** `data/raw/kenpom/kenpom_2026.csv`
- **Source:** kenpom.com → summary table
- **Columns:** `team_name, adj_em, adj_t, luck`
- **Note:** `adj_em` = adjusted efficiency margin, `adj_t` = adjusted tempo, `luck` = KenPom luck rating
- **Action:** User copies from KenPom and pastes into this CSV

### 4. Kaggle Historical Games — Automated download
- **Source:** Kaggle March Machine Learning Mania competition dataset
- **Files downloaded:**
  - `MRegularSeasonCompactResults.csv` — historical game results (WTeamID, LTeamID, WScore, LScore, Season)
  - `MTeams.csv` — team ID to team name lookup
- **Processed output:** `data/raw/kaggle/regular_season_2026.csv`
- **Credentials:** `~/.kaggle/kaggle.json` (one-time setup, see SETUP.md)
- **Action:** User runs `python scripts/download_data.py`

---

## Key Implementation Issues to Resolve

### Crosswalk: Fake IDs → Real Kaggle IDs

The existing `build_crosswalk.py` assigns sequential integers (1, 2, 3…) as `kaggle_team_id`, which do not match actual Kaggle team IDs (e.g., 1101, 1102…). This must be fixed:

`scripts/download_data.py` will:
1. Download the competition dataset via Kaggle CLI
2. Read `MTeams.csv` to build a `TeamName → TeamID` lookup
3. Produce `data/raw/kaggle/regular_season_2026.csv` with real numeric Kaggle team IDs

`src/data/build_crosswalk.py` will be updated to:
1. Read the downloaded `MTeams.csv`
2. Fuzzy-match Barttorvik team names → Kaggle team names
3. Assign real `kaggle_team_id` values instead of sequential integers

### KenPom Columns

The existing `ingest_kenpom.py` requires exactly: `team_name, adj_em, adj_t, luck`. The CSV template must match these column names.

### Kaggle Package

Add `kaggle>=1.6` to `requirements.txt` so the download script can use the Kaggle API.

---

## Changes Required

### New files
| File | Purpose |
|------|---------|
| `scripts/download_data.py` | Kaggle CLI download + preprocessing script |
| `data/raw/bracket/bracket_2026.csv` | Pre-populated 2026 bracket (68 teams) |
| `data/raw/barttorvik/barttorvik_2026.csv` | CSV template with headers for manual paste |
| `data/raw/kenpom/kenpom_2026.csv` | CSV template with headers for manual paste |
| `SETUP.md` | Kaggle credentials setup instructions |

### Modified files
| File | Change |
|------|--------|
| `configs/config.yaml` | Set `local_path` for all four data sources; `num_teams: 68` |
| `requirements.txt` | Add `kaggle>=1.6` |
| `src/data/build_crosswalk.py` | Use real Kaggle MTeams.csv for name→ID matching |

---

## Command Sequence

```bash
# One-time setup: place kaggle.json at ~/.kaggle/kaggle.json (see SETUP.md)
pip install -r requirements.txt

# Step 1: Download Kaggle historical data + team lookup
python scripts/download_data.py

# Step 2: Paste Barttorvik Team Stats into:
#   data/raw/barttorvik/barttorvik_2026.csv
#   Columns: team_name, adj_o, adj_d, tempo

# Step 3: Paste KenPom stats into:
#   data/raw/kenpom/kenpom_2026.csv
#   Columns: team_name, adj_em, adj_t, luck

# Step 4: Run the full pipeline
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

---

## 2026 Bracket (68 Teams)

### East Region
| Seed | Team | Slot |
|------|------|------|
| 1 | Duke | E1 |
| 16 | Siena | E16 |
| 8 | Ohio St. | E8 |
| 9 | TCU | E9 |
| 5 | St. John's | E5 |
| 12 | N. Iowa | E12 |
| 4 | Kansas | E4 |
| 13 | Cal Baptist | E13 |
| 6 | Louisville | E6 |
| 11 | South Florida | E11 |
| 3 | Michigan St. | E3 |
| 14 | N. Dakota St. | E14 |
| 7 | UCLA | E7 |
| 10 | UCF | E10 |
| 2 | UConn | E2 |
| 15 | Furman | E15 |

### South Region
| Seed | Team | Slot |
|------|------|------|
| 1 | Florida | S1 |
| 16 | Prairie View A&M | S16 |
| 8 | Clemson | S8 |
| 9 | Iowa | S9 |
| 5 | Vanderbilt | S5 |
| 12 | McNeese | S12 |
| 4 | Nebraska | S4 |
| 13 | Troy | S13 |
| 6 | North Carolina | S6 |
| 11 | VCU | S11 |
| 3 | Illinois | S3 |
| 14 | Penn | S14 |
| 7 | Saint Mary's | S7 |
| 10 | Texas A&M | S10 |
| 2 | Houston | S2 |
| 15 | Idaho | S15 |

### West Region
| Seed | Team | Slot |
|------|------|------|
| 1 | Arizona | W1 |
| 16 | LIU | W16 |
| 8 | Villanova | W8 |
| 9 | Utah St. | W9 |
| 5 | Wisconsin | W5 |
| 12 | High Point | W12 |
| 4 | Arkansas | W4 |
| 13 | Hawaii | W13 |
| 6 | BYU | W6 |
| 11 | Texas | W11 |
| 3 | Gonzaga | W3 |
| 14 | Kennesaw St. | W14 |
| 7 | Miami | W7 |
| 10 | Missouri | W10 |
| 2 | Purdue | W2 |
| 15 | Queens | W15 |

### Midwest Region
| Seed | Team | Slot |
|------|------|------|
| 1 | Michigan | M1 |
| 16 | UMBC | M16 |
| 8 | Georgia | M8 |
| 9 | Saint Louis | M9 |
| 5 | Texas Tech | M5 |
| 12 | Akron | M12 |
| 4 | Alabama | M4 |
| 13 | Hofstra | M13 |
| 6 | Tennessee | M6 |
| 11 | SMU | M11 |
| 3 | Virginia | M3 |
| 14 | Wright St. | M14 |
| 7 | Kentucky | M7 |
| 10 | Santa Clara | M10 |
| 2 | Iowa St. | M2 |
| 15 | Tennessee St. | M15 |

### First Four (March 17-18)
| Slot | Game |
|------|------|
| S16 | Prairie View A&M vs Lehigh |
| W11 | Texas vs NC State |
| M16 | UMBC vs Howard |
| M11 | Miami (OH) vs SMU |

---

## Notes
- The bracket CSV uses First Four winners as placeholders; after First Four games play (March 17-18), users may update the bracket CSV with actual winners before running the pipeline
- Team names in Barttorvik and KenPom CSVs must match the bracket names closely enough for the crosswalk fuzzy match to resolve them
- Kaggle competition: `march-machine-learning-mania-2026`
- `num_teams` in config updated to 68 to match the actual field size
