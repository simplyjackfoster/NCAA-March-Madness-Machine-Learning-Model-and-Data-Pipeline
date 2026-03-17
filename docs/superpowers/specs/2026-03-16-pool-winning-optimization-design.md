# Pool-Winning Optimization Design

## Goal

Maximize P(win office pool) for a 16-24 person ESPN-style March Madness bracket pool by fixing broken pipeline components and adding a Monte Carlo pool simulator.

## Context

Current pipeline has three critical flaws that make its pool recommendations unreliable:
1. ESPN pick rates are derived from the model's own output (circular — leverage is always ~1.0)
2. Probabilities are uncalibrated (nearly all matchups show 100% — elo_diff percentile is used instead of actual model probabilities)
3. pool_size in config is 50; actual pool is 16-24 people

KenPom data is ingested but never used. Three noisy warnings flood pipeline output.

## Scoring System

ESPN standard: 1 / 2 / 4 / 8 / 16 / 32 points per round (rounds 1–6).

## Pool Size

Medium: 16-24 people. Config will use `pool_size: 20` as the midpoint.

## Architecture

```
Features  →  Model  →  Calibration  →  Tournament Sim  →  Pool Sim  →  Champion Pick
```

### Data Flow

```
barttorvik_2026.csv  ─┐
kenpom_2026.csv      ─┼─→  team_features (+ seed, adj_em, luck)
bracket_2026.csv     ─┘         │
                                 ▼
kaggle historical  ──────→  game_features (+ seed_diff)
                                 │
                                 ▼
                          ensemble model
                                 │
                                 ▼
                     Platt calibration  ←── train on historical tourney outcomes
                                 │
                                 ▼
                          prob_matrix_{year}.npy
                                 │
                          ┌──────┴──────┐
                          ▼             ▼
                   tourney sim     seed_popularity model
                   (5000 sims)     (historical pick rates by seed)
                        │               │
                        └──────┬────────┘
                               ▼
                        pool_simulator
                        (10k pool sims × N opponents)
                               │
                               ▼
                    top-3 champion picks
                    + P(win pool) for each
```

## Component Details

### New: `src/field/seed_popularity.py`

Hard-coded historical championship pick rates by seed derived from ESPN Tournament Challenge data (2013–2024):

| Seed | Championship pick % |
|------|-------------------|
| 1    | ~58% (split across 4 teams: ~14.5% each) |
| 2    | ~25% (split across 4 teams: ~6.25% each) |
| 3    | ~10% (split across 4 teams: ~2.5% each) |
| 4–5  | ~5% combined |
| 6+   | ~2% combined |

Name-recognition boost applied to historically popular programs: Duke, Kentucky, Kansas, North Carolina, Michigan (each +1-2%).

Returns `{team_name: pick_pct}` normalized to sum to 1.0.

### New: `src/simulation/pool_simulator.py`

Inputs:
- `prob_matrix` — calibrated n×n win probability matrix
- `seed_popularity` — `{team: field_pick_pct}` from seed model
- `bracket_order` — list of team indices in slot order
- `pool_size` — number of opponents (default 20)
- `num_sims` — tournament simulations (default 10,000)
- `scoring` — points per round [1, 2, 4, 8, 16, 32]

Algorithm:
1. For each tournament simulation:
   a. Simulate one tournament outcome using prob_matrix → compute scores for all possible brackets
   b. Generate `pool_size` synthetic opponent brackets by sampling champion picks from seed_popularity, then filling remaining picks proportional to prob_matrix
   c. Score each opponent bracket against the tournament outcome
2. For each candidate champion pick, track how often the user's bracket score exceeds all opponents' scores
3. Return `{team: p_win_pool}` sorted descending

### Modified: `src/field/espn_loader.py`

Replace circular formula (`model_prob * 1.3 + 0.01`) with output from `seed_popularity.py`. Existing interface preserved — still returns a path to a JSON file.

### Modified: `src/features/team_features.py`

Add three columns:
- `seed` — from `bracket_{year}.csv` (join on team_name)
- `adj_em` — from KenPom (`adj_em = adj_o - adj_d` proxy if not present, or direct column)
- `luck` — from KenPom (0.0 default if not present)

### Modified: `src/features/game_features.py`

Add `seed_diff` feature to matchup feature vector. Seeds are strong predictors of tournament outcomes independent of efficiency ratings.

### Modified: `src/models/calibrate.py`

Replace elo-percentile hack with Platt scaling (logistic regression) fit on actual ensemble model output probabilities vs historical tournament game outcomes. This produces realistic win probabilities (55%/60% range) instead of near-0%/100%.

### Modified: `src/optimization/leverage.py`

Replace `leverage = model_prob / field_prob` formula with P(win pool) output from `pool_simulator.py`. Pool-size-aware by construction.

### Modified: `configs/config.yaml`

```yaml
optimization:
  pool_size: 20
  pool_sims: 10000
  min_champion_prob: 0.02
  risk_tolerance: balanced
```

## Warning Fixes

Three warnings currently flood pipeline output on every run:

1. **XGBoost `use_label_encoder`** — remove deprecated parameter from `src/models/xgb_model.py`
2. **numpy zero-variance divide** — add `.clip(lower=1e-9)` in `src/models/ensemble.py` correlation computation
3. **sklearn feature names** — pass DataFrame with column names (not raw numpy arrays) through prediction paths in `src/simulation/matchup_matrix.py`

## Output Changes

Current: single champion recommendation.

New: top-3 champion picks displayed in bracket output:
```
Top Champion Picks for Pool (pool size: 20)
==========================================
1. Duke          model_prob=38%  field_pick=18%  P(win pool)=12.4%
2. Florida       model_prob=22%  field_pick=8%   P(win pool)=9.1%
3. Michigan      model_prob=18%  field_pick=12%  P(win pool)=6.8%
```

## Testing Approach

- Unit test `seed_popularity.py`: pick rates sum to 1.0, all 64 bracket teams get a non-zero rate
- Unit test `pool_simulator.py`: with a 2-team bracket and known probs, verify P(win pool) is analytically correct
- Integration test: run full pipeline with 2024 data (known outcomes), verify calibrated probs are in [0.45, 0.85] range for seed-1 vs seed-16
- Regression test: existing `bracket_2026.txt` output format unchanged

## Files Summary

| File | Action |
|------|--------|
| `src/field/seed_popularity.py` | Create |
| `src/simulation/pool_simulator.py` | Create |
| `src/field/espn_loader.py` | Modify |
| `src/features/team_features.py` | Modify |
| `src/features/game_features.py` | Modify |
| `src/models/calibrate.py` | Modify |
| `src/optimization/leverage.py` | Modify |
| `src/models/xgb_model.py` | Modify (warning fix) |
| `src/models/ensemble.py` | Modify (warning fix) |
| `src/simulation/matchup_matrix.py` | Modify (warning fix) |
| `configs/config.yaml` | Modify |
