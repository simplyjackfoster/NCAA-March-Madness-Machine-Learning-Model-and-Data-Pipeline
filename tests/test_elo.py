from src.features.elo import expected_score


def test_expected_score_midpoint():
    assert abs(expected_score(1500, 1500) - 0.5) < 1e-9
