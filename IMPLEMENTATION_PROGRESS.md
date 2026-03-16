# NCAA Office Pool Pipeline — Implementation Progress Tracker

This file tracks implementation status against `NCAA_OfficePool_Engineering_Blueprint.docx`.

## How to use this tracker
- Keep statuses up to date after each merged PR.
- Use one of: `Not Started`, `In Progress`, `Done`, `Deferred`.
- Add PR/commit references in the Notes column.

---

## High-level phase status

| Phase | Blueprint Goal | Status | Notes |
|---|---|---|---|
| Phase 1 | Basic probability engine | In Progress | Core pipeline exists; validation is still simplified. |
| Phase 2 | Calibration + extended features + matchup matrix | In Progress | Calibrator and matchup matrix exist; diagnostics/extended ingestion pending. |
| Phase 3 | Monte Carlo tournament simulation | In Progress | Simulation artifacts produced, but not full score simulation module parity. |
| Phase 4 | Field model | In Progress | Pool model exists but still uses ESPN placeholder behavior. |
| Phase 5 | Bracket optimization | In Progress | Champion search + greedy selection exist; expected-score/leverage modules pending. |
| Phase 6 | Export + sanity checks | In Progress | Bracket text export exists; strategy report module pending. |

---

## Module/file checklist (from blueprint)

### Data

| Module | Status | Notes |
|---|---|---|
| `src/data/ingest_kaggle.py` | Not Started | Not present in repo. |
| `src/data/ingest_barttorvik.py` | Done | Present and wired in pipeline. |
| `src/data/ingest_kenpom.py` | Not Started | Not present in repo. |
| `src/data/build_crosswalk.py` | Done | Present and wired in pipeline. |

### Features

| Module | Status | Notes |
|---|---|---|
| `src/features/team_features.py` | Done | Present and wired in pipeline. |
| `src/features/game_features.py` | Done | Present and wired in pipeline. |
| `src/features/elo.py` | Done | Present. |
| `src/features/feature_registry.py` | Not Started | Not present in repo. |

### Models

| Module | Status | Notes |
|---|---|---|
| `src/models/prior_model.py` | Done | Present and wired in pipeline. |
| `src/models/lgbm_model.py` | Done | Present and wired in pipeline. |
| `src/models/xgb_model.py` | Done | Present and wired in pipeline. |
| `src/models/ensemble.py` | Not Started | Not present in repo. |
| `src/models/loyo_validator.py` | In Progress | Present, but currently pseudo-LOYO split fallback. |

### Calibration

| Module | Status | Notes |
|---|---|---|
| `src/models/calibrate.py` | In Progress | Present; diagnostics/report integration still pending. |
| `src/models/diagnostics.py` | Not Started | Not present in repo. |

### Simulation

| Module | Status | Notes |
|---|---|---|
| `src/simulation/matchup_matrix.py` | Done | Present and wired in pipeline. |
| `src/simulation/tournament_sim.py` | Done | Present and wired in pipeline. |
| `src/simulation/score_sim.py` | Not Started | Not present in repo. |

### Field model

| Module | Status | Notes |
|---|---|---|
| `src/field/espn_loader.py` | Not Started | Not present in repo. |
| `src/field/pool_model.py` | In Progress | Present, uses ESPN placeholder logic. |
| `src/field/field_sampler.py` | Not Started | Not present in repo. |

### Optimization

| Module | Status | Notes |
|---|---|---|
| `src/optimization/expected_score.py` | Not Started | Not present in repo. |
| `src/optimization/leverage.py` | Not Started | Not present in repo. |
| `src/optimization/champion_search.py` | In Progress | Present; simplified P(win pool) proxy. |
| `src/optimization/greedy_optimizer.py` | Done | Present and wired in pipeline. |
| `src/optimization/annealing.py` | Not Started | Not present in repo. |

### Export

| Module | Status | Notes |
|---|---|---|
| `src/export/bracket_formatter.py` | Done | Present and wired in pipeline. |
| `src/export/strategy_report.py` | Not Started | Not present in repo. |

### Script entrypoints (blueprint `scripts/*.py`)

| Script | Status | Notes |
|---|---|---|
| `scripts/ingest_data.py` | Not Started | Directory not present in repo. |
| `scripts/build_features.py` | Not Started | Directory not present in repo. |
| `scripts/train_models.py` | Not Started | Directory not present in repo. |
| `scripts/calibrate_models.py` | Not Started | Directory not present in repo. |
| `scripts/generate_matchups.py` | Not Started | Directory not present in repo. |
| `scripts/run_simulations.py` | Not Started | Directory not present in repo. |
| `scripts/build_field_model.py` | Not Started | Directory not present in repo. |
| `scripts/optimize_brackets.py` | Not Started | Directory not present in repo. |
| `scripts/export_bracket.py` | Not Started | Directory not present in repo. |

### Tests (blueprint suggestions)

| Test file | Status | Notes |
|---|---|---|
| `tests/test_features.py` | Not Started | `tests/` folder not present. |
| `tests/test_elo.py` | Not Started | `tests/` folder not present. |
| `tests/test_simulation.py` | Not Started | `tests/` folder not present. |
| `tests/test_optimization.py` | Not Started | `tests/` folder not present. |

---

## Current next priorities
1. Implement true year-based LOYO in `src/models/loyo_validator.py`.
2. Add calibration diagnostics (`ECE`, reliability output) module.
3. Add `espn_loader.py` + replace placeholder behavior in `pool_model.py`.
4. Add `score_sim.py` and wire score distribution artifact generation.
5. Add missing script entrypoints under `scripts/` for stage-by-stage execution.

## Change log
- 2026-03-16: Created tracker file with phase/module/script/test status baseline.
