import pandas as pd
import numpy as np

from src.features.feature_registry import get_feature_names


def test_feature_registry_has_core_features():
    names = get_feature_names()
    assert "elo_diff" in names
    assert "net_rating" in names


def test_team_features_includes_kenpom_and_seed(tmp_path, monkeypatch):
    """team_season parquet must include adj_em, luck, and seed columns."""
    import yaml
    from src.common import config as config_module

    # Write minimal config
    config = {
        "project": {"target_year": 2025, "base_data_dir": str(tmp_path / "data"),
                     "artifacts_dir": str(tmp_path / "artifacts"), "outputs_dir": str(tmp_path / "outputs")},
        "data": {"random_seed": 42, "num_teams": 4},
        "data_sources": {},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    # Patch load_config to use tmp_path as root
    def mock_load_config(config_path: str = "configs/config.yaml"):
        cfg = yaml.safe_load(cfg_path.open("r"))
        cfg["_root"] = tmp_path
        return cfg

    monkeypatch.setattr(config_module, "load_config", mock_load_config)

    year = 2025
    # Write barttorvik CSV
    bt_dir = tmp_path / "data" / "raw" / "barttorvik"
    bt_dir.mkdir(parents=True)
    pd.DataFrame({
        "season": [year] * 4,
        "team_name": ["TeamA", "TeamB", "TeamC", "TeamD"],
        "adj_o": [110.0, 105.0, 100.0, 95.0],
        "adj_d": [90.0, 95.0, 100.0, 105.0],
        "tempo": [70.0, 68.0, 66.0, 64.0],
    }).to_csv(bt_dir / f"barttorvik_{year}.csv", index=False)

    # Write crosswalk
    cx_dir = tmp_path / "data" / "crosswalks"
    cx_dir.mkdir(parents=True)
    pd.DataFrame({
        "barttorvik_name": ["TeamA", "TeamB", "TeamC", "TeamD"],
        "kaggle_team_id": [1, 2, 3, 4],
        "display_name": ["TeamA", "TeamB", "TeamC", "TeamD"],
    }).to_csv(cx_dir / "team_id_map.csv", index=False)

    # Write kenpom CSV
    kp_dir = tmp_path / "data" / "raw" / "kenpom"
    kp_dir.mkdir(parents=True)
    pd.DataFrame({
        "season": [year] * 4,
        "team_name": ["TeamA", "TeamB", "TeamC", "TeamD"],
        "adj_em": [20.0, 15.0, 5.0, -5.0],
        "adj_t": [70.0, 68.0, 66.0, 64.0],
        "luck": [0.02, -0.01, 0.00, 0.03],
    }).to_csv(kp_dir / f"kenpom_{year}.csv", index=False)

    # Write bracket CSV with seeds
    br_dir = tmp_path / "data" / "raw" / "bracket"
    br_dir.mkdir(parents=True)
    pd.DataFrame({
        "team_name": ["TeamA", "TeamB", "TeamC", "TeamD"],
        "seed": [1, 8, 5, 4],
        "region": ["East"] * 4,
        "slot": [1, 2, 3, 4],
    }).to_csv(br_dir / f"bracket_{year}.csv", index=False)

    # Import after patching
    from src.features.team_features import build_team_features

    out = build_team_features(year, "dummy_config")
    df = pd.read_parquet(out)

    assert "adj_em" in df.columns, "adj_em missing"
    assert "luck" in df.columns, "luck missing"
    assert "seed" in df.columns, "seed missing"
    assert df.loc[df["display_name"] == "TeamA", "seed"].iloc[0] == 1
    assert df.loc[df["display_name"] == "TeamA", "adj_em"].iloc[0] == 20.0
    assert not df["adj_em"].isna().any()
    assert not df["seed"].isna().any()


def test_game_features_includes_seed_diff(tmp_path):
    """games parquet must include seed_diff column."""
    import yaml
    from src.features.game_features import build_game_features

    config = {
        "project": {"target_year": 2025, "base_data_dir": str(tmp_path / "data"),
                     "artifacts_dir": str(tmp_path / "artifacts"), "outputs_dir": str(tmp_path / "outputs")},
        "data": {"random_seed": 42, "num_teams": 4},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    year = 2025
    feat_dir = tmp_path / "data" / "features"
    feat_dir.mkdir(parents=True)
    pd.DataFrame({
        "season": [year] * 4,
        "kaggle_team_id": [1, 2, 3, 4],
        "display_name": ["TeamA", "TeamB", "TeamC", "TeamD"],
        "adj_o": [110.0, 105.0, 100.0, 95.0],
        "adj_d": [90.0, 95.0, 100.0, 105.0],
        "tempo": [70.0, 68.0, 66.0, 64.0],
        "net_rating": [20.0, 10.0, 0.0, -10.0],
        "elo_pre": [1660.0, 1580.0, 1500.0, 1420.0],
        "adj_em": [20.0, 10.0, 0.0, -10.0],
        "luck": [0.0] * 4,
        "seed": [1, 8, 5, 4],
    }).to_parquet(feat_dir / f"team_season_{year}.parquet", index=False)

    out = build_game_features(year, str(cfg_path))
    df = pd.read_parquet(out)
    assert "seed_diff" in df.columns, "seed_diff missing from game features"


def test_build_calibration_set_produces_valid_probs(tmp_path):
    """calibration_set.parquet must have model_prob in (0, 1) and outcome in {0, 1}."""
    import yaml
    from src.data.build_calibration_set import build_calibration_set

    config = {
        "project": {"target_year": 2025, "base_data_dir": str(tmp_path / "data"),
                     "artifacts_dir": str(tmp_path / "artifacts"), "outputs_dir": str(tmp_path / "outputs")},
        "data": {"random_seed": 42, "num_teams": 4},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    # Write synthetic training data
    proc_dir = tmp_path / "data" / "processed"
    proc_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)
    n = 200
    elo = rng.normal(0, 5, n)
    pd.DataFrame({
        "elo_diff": elo,
        "net_rating_diff": elo * 0.5 + rng.normal(0, 2, n),
        "tempo_diff": rng.normal(0, 1, n),
        "seed_diff": rng.integers(-15, 15, n),
        "label": (elo + rng.normal(0, 2, n) > 0).astype(int),
    }).to_parquet(proc_dir / "train.parquet", index=False)

    out = build_calibration_set(str(cfg_path))
    df = pd.read_parquet(out)

    assert "model_prob" in df.columns
    assert "outcome" in df.columns
    assert df["model_prob"].between(0, 1, inclusive="neither").all(), "model_prob must be in open interval (0, 1)"
    assert set(df["outcome"].unique()).issubset({0, 1})
    assert len(df) == n
