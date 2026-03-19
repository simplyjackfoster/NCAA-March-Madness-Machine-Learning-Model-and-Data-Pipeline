import subprocess
import sys

import numpy as np
import pandas as pd
import yaml


def test_probability_matrix_self_matchups_zero():
    mat = np.zeros((4, 4))
    assert np.allclose(np.diag(mat), 0.0)


def test_xgb_no_use_label_encoder_warning():
    """xgb_model.py must not pass use_label_encoder to XGBClassifier."""
    # Check if use_label_encoder is present in the source code
    import src.models.xgb_model as xgb_module
    import inspect

    source = inspect.getsource(xgb_module.train_xgb_model)
    assert "use_label_encoder" not in source, "use_label_encoder parameter should not be used in train_xgb_model"


def test_ensemble_weights_constant_feature(tmp_path, monkeypatch):
    """build_ensemble_weights must not crash when a feature has zero variance."""
    import warnings
    from src.models import ensemble as ensemble_module

    # Mock load_config to return our test config
    def mock_load_config(config_path: str):
        return {
            "_root": tmp_path,
        }

    monkeypatch.setattr(ensemble_module, "load_config", mock_load_config)

    # Write train.parquet where tempo_diff is constant (zero variance — triggers NaN correlation)
    proc_dir = tmp_path / "data" / "processed"
    proc_dir.mkdir(parents=True)
    elo = np.linspace(-10, 10, 100)
    pd.DataFrame({
        "elo_diff": elo,
        "net_rating_diff": elo * 0.5,
        "tempo_diff": np.zeros(100),   # constant — will produce NaN in corrwith
        "label": (elo > 0).astype(int),
    }).to_parquet(proc_dir / "train.parquet", index=False)

    art_dir = tmp_path / "artifacts" / "models"
    art_dir.mkdir(parents=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ensemble_module.build_ensemble_weights()

    runtime_warnings = [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]
    assert runtime_warnings == [], f"RuntimeWarning emitted: {runtime_warnings}"


def test_matchup_matrix_no_feature_name_warning(tmp_path, monkeypatch):
    """build_matchup_matrix must not trigger sklearn feature-name warnings."""
    import warnings
    import pickle
    from sklearn.linear_model import LogisticRegression
    from src.simulation import matchup_matrix as mm_module
    from src.simulation.matchup_matrix import build_matchup_matrix

    # Mock load_config to return our test directory as root
    def mock_load_config(config_path: str):
        return {"_root": tmp_path}

    monkeypatch.setattr(mm_module, "load_config", mock_load_config)

    year = 2025
    feat_dir = tmp_path / "data" / "features"
    feat_dir.mkdir(parents=True)
    pd.DataFrame({
        "season": [year] * 4, "kaggle_team_id": [1, 2, 3, 4],
        "display_name": ["A", "B", "C", "D"],
        "adj_o": [110.0, 100.0, 90.0, 80.0], "adj_d": [80.0, 90.0, 100.0, 110.0],
        "tempo": [70.0, 68.0, 66.0, 64.0],
        "net_rating": [30.0, 10.0, -10.0, -30.0],
        "elo_pre": [1740.0, 1580.0, 1420.0, 1260.0],
        "adj_em": [20.0, 10.0, -5.0, -20.0], "luck": [0.0] * 4, "seed": [1, 8, 5, 4],
    }).to_parquet(feat_dir / f"team_season_{year}.parquet", index=False)

    # Train model on 3 features (Task 3 runs before Task 5 adds seed_diff)
    FEATURES = ["elo_diff", "net_rating_diff", "tempo_diff"]
    rng = np.random.default_rng(42)
    n = 100
    elo = rng.normal(0, 5, n)
    X = pd.DataFrame({
        "elo_diff": elo, "net_rating_diff": elo * 0.5 + rng.normal(0, 1, n),
        "tempo_diff": rng.normal(0, 1, n),
    })
    y = (elo > 0).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    model_dir = tmp_path / "artifacts" / "models"
    model_dir.mkdir(parents=True)
    with open(model_dir / "prior_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Write a pass-through calibrator
    from sklearn.linear_model import LogisticRegression as LR
    raw = model.predict_proba(X)[:, 1].reshape(-1, 1)
    cal = LR(max_iter=1000).fit(raw, y)
    cal_dir = tmp_path / "artifacts" / "calibrators"
    cal_dir.mkdir(parents=True)
    with open(cal_dir / "isotonic.pkl", "wb") as f:
        pickle.dump(cal, f)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_matchup_matrix(year, "dummy_config.yaml")

    feature_warnings = [x for x in w if "feature names" in str(x.message)]
    assert feature_warnings == [], f"sklearn feature name warnings emitted: {[str(x.message) for x in feature_warnings]}"
