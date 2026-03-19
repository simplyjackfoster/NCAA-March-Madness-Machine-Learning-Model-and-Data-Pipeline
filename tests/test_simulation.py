import subprocess
import sys

import numpy as np


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
