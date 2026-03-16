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
| Phase 1 | Basic probability engine | Done | Core probability engine and true LOYO validator are implemented. |
| Phase 2 | Calibration + extended features + matchup matrix | Done | Calibration diagnostics and extended ingestion modules are implemented. |
| Phase 3 | Monte Carlo tournament simulation | Done | Tournament simulation plus score distribution module are implemented. |
| Phase 4 | Field model | Done | ESPN loader and field sampler are implemented and wired. |
| Phase 5 | Bracket optimization | Done | Expected-score, leverage, annealing modules are implemented. |
| Phase 6 | Export + sanity checks | Done | Strategy report module and script entrypoints are implemented. |

---

## Module/file checklist (from blueprint)

### Data

| Module | Status | Notes |
|---|---|---|
| `src/data/ingest_kaggle.py` | Done | Implemented offline-friendly ingest scaffold. |
| `src/data/ingest_barttorvik.py` | Done | Present and wired in pipeline. |
| `src/data/ingest_kenpom.py` | Done | Implemented offline-friendly ingest scaffold. |
| `src/data/build_crosswalk.py` | Done | Present and wired in pipeline. |

### Features

| Module | Status | Notes |
|---|---|---|
| `src/features/team_features.py` | Done | Present and wired in pipeline. |
| `src/features/game_features.py` | Done | Present and wired in pipeline. |
| `src/features/elo.py` | Done | Present. |
| `src/features/feature_registry.py` | Done | Implemented with helper accessors for stage-specific features. |

### Models

| Module | Status | Notes |
|---|---|---|
| `src/models/prior_model.py` | Done | Present and wired in pipeline. |
| `src/models/lgbm_model.py` | Done | Present and wired in pipeline. |
| `src/models/xgb_model.py` | Done | Present and wired in pipeline. |
| `src/models/ensemble.py` | Done | Implemented deterministic ensemble-weight artifact generation. |
| `src/models/loyo_validator.py` | Done | Updated to true leave-one-year-out folds. |

### Calibration

| Module | Status | Notes |
|---|---|---|
| `src/models/calibrate.py` | Done | Diagnostics integration implemented. |
| `src/models/diagnostics.py` | Done | Implemented ECE + reliability outputs. |

### Simulation

| Module | Status | Notes |
|---|---|---|
| `src/simulation/matchup_matrix.py` | Done | Present and wired in pipeline. |
| `src/simulation/tournament_sim.py` | Done | Present and wired in pipeline. |
| `src/simulation/score_sim.py` | Done | Implemented score distribution artifact generation. |

### Field model

| Module | Status | Notes |
|---|---|---|
| `src/field/espn_loader.py` | Done | Implemented offline ESPN pick-rate loader/proxy. |
| `src/field/pool_model.py` | Done | Wired to espn_loader module. |
| `src/field/field_sampler.py` | Done | Implemented field bracket sampler. |

### Optimization

| Module | Status | Notes |
|---|---|---|
| `src/optimization/expected_score.py` | Done | Implemented expected-score module. |
| `src/optimization/leverage.py` | Done | Implemented leverage module. |
| `src/optimization/champion_search.py` | Done | Kept proxy method and integrated downstream modules. |
| `src/optimization/greedy_optimizer.py` | Done | Present and wired in pipeline. |
| `src/optimization/annealing.py` | Done | Implemented annealing-style final selection stub. |

### Export

| Module | Status | Notes |
|---|---|---|
| `src/export/bracket_formatter.py` | Done | Present and wired in pipeline. |
| `src/export/strategy_report.py` | Done | Implemented strategy report export module. |

### Script entrypoints (blueprint `scripts/*.py`)

| Script | Status | Notes |
|---|---|---|
| `scripts/ingest_data.py` | Done | Added script entrypoint. |
| `scripts/build_features.py` | Done | Added script entrypoint. |
| `scripts/train_models.py` | Done | Added script entrypoint. |
| `scripts/calibrate_models.py` | Done | Added script entrypoint. |
| `scripts/generate_matchups.py` | Done | Added script entrypoint. |
| `scripts/run_simulations.py` | Done | Added script entrypoint. |
| `scripts/build_field_model.py` | Done | Added script entrypoint. |
| `scripts/optimize_brackets.py` | Done | Added script entrypoint. |
| `scripts/export_bracket.py` | Done | Added script entrypoint. |

### Tests (blueprint suggestions)

| Test file | Status | Notes |
|---|---|---|
| `tests/test_features.py` | Done | Added baseline registry test. |
| `tests/test_elo.py` | Done | Added Elo expectation test. |
| `tests/test_simulation.py` | Done | Added baseline simulation test. |
| `tests/test_optimization.py` | Done | Added baseline optimization test. |

---

## Current next priorities
1. Improve champion search objective beyond current proxy for direct P(win pool).
2. Expand field model to support optional live ESPN ingestion endpoint/file feed.
3. Add richer bracket-level simulation diagnostics and confidence intervals.
4. Add integration tests for end-to-end stage scripts.

## Change log
- 2026-03-16: Created tracker file with phase/module/script/test status baseline.
- 2026-03-16: Implemented missing modules, scripts, and baseline tests from checklist.
