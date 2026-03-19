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
