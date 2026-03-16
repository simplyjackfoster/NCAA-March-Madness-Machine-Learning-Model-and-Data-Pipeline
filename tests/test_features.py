from src.features.feature_registry import get_feature_names


def test_feature_registry_has_core_features():
    names = get_feature_names()
    assert "elo_diff" in names
    assert "net_rating" in names
