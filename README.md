# NCAA March Madness Office Pool ML Pipeline

Implementation of the engineering blueprint in `NCAA_OfficePool_Engineering_Blueprint.docx` as a runnable, file-based pipeline.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py --year 2026
```

## Pipeline outputs

- Raw input scaffold: `data/raw/barttorvik/barttorvik_{year}.csv`
- Feature sets: `data/features/team_season_{year}.parquet`, `data/features/games_{year}.parquet`
- Models: `artifacts/models/prior_model.pkl`, `artifacts/models/lgbm_model.pkl`, `artifacts/models/xgb_model.pkl`
- Calibration + validation: `artifacts/calibrators/isotonic.pkl`, `outputs/validation/loyo_results.json`
- Tournament simulation: `data/tournament/prob_matrix_{year}.npy`, `data/simulation/advance_probs_{year}.json`
- Optimization + output: `data/optimization/candidates_{year}.parquet`, `outputs/bracket_{year}_final.json`, `outputs/bracket_{year}.txt`

## Notes

- This implementation is intentionally MVP-oriented and offline-friendly.
- Stage scripts are modular and can be run independently.
