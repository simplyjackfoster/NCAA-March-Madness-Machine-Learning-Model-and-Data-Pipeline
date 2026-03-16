import numpy as np


def test_probability_matrix_self_matchups_zero():
    mat = np.zeros((4, 4))
    assert np.allclose(np.diag(mat), 0.0)
