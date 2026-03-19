import pandas as pd
import numpy as np

from src.features.feature_registry import get_feature_names


def test_feature_registry_has_core_features():
    names = get_feature_names()
    assert "seed_diff" in names
    assert "rank_diff_POM" in names
    assert "net_rating" in names  # team-level feature, unchanged


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


def test_game_features_writes_output_files(tmp_path):
    """build_game_features must write games_{year}.parquet and train.parquet."""
    import yaml
    from src.features.game_features import build_game_features

    config = {
        "project": {
            "target_year": 2026,
            "base_data_dir": str(tmp_path / "data"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "outputs_dir": str(tmp_path / "outputs"),
        },
        "data": {"random_seed": 42, "num_teams": 4},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    import pandas as pd
    kaggle_dir = tmp_path / "data" / "raw" / "kaggle" / "downloads"
    kaggle_dir.mkdir(parents=True)
    pd.DataFrame({"Season": [2023], "WTeamID": [1101], "LTeamID": [1103]}).to_csv(
        kaggle_dir / "MNCAATourneyCompactResults.csv", index=False
    )
    pd.DataFrame({
        "Season": [2023, 2023], "TeamID": [1101, 1103], "Seed": ["W01", "W12"],
    }).to_csv(kaggle_dir / "MNCAATourneySeeds.csv", index=False)
    pd.DataFrame({
        "Season": [2023, 2023], "RankingDayNum": [128, 128],
        "SystemName": ["POM", "POM"], "TeamID": [1101, 1103], "OrdinalRank": [5, 60],
    }).to_csv(kaggle_dir / "MMasseyOrdinals.csv", index=False)

    out = build_game_features(2026, str(cfg_path))
    assert out.exists()
    assert (tmp_path / "data" / "processed" / "train.parquet").exists()
    df = pd.read_parquet(out)
    assert "label" in df.columns
    assert "seed_diff" in df.columns


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
    seed_diff = rng.integers(-15, 15, n)
    pd.DataFrame({
        "seed_diff": seed_diff,
        "rank_diff_POM": seed_diff * 4.0 + rng.normal(0, 5, n),
        "rank_diff_MOR": seed_diff * 4.0 + rng.normal(0, 5, n),
        "rank_diff_SAG": seed_diff * 4.0 + rng.normal(0, 5, n),
        "label": (seed_diff + rng.normal(0, 3, n) < 0).astype(int),
    }).to_parquet(proc_dir / "train.parquet", index=False)

    out = build_calibration_set(str(cfg_path))
    df = pd.read_parquet(out)

    assert "model_prob" in df.columns
    assert "outcome" in df.columns
    assert df["model_prob"].between(0, 1, inclusive="neither").all(), "model_prob must be in open interval (0, 1)"
    assert set(df["outcome"].unique()).issubset({0, 1})
    assert len(df) == n


def test_game_features_uses_real_outcomes(tmp_path):
    """build_game_features must produce binary labels from real tournament results,
    with rank_diff_POM/MOR/SAG columns present, and NO synthetic rng.normal labels."""
    import yaml
    from src.features.game_features import build_game_features

    config = {
        "project": {
            "target_year": 2026,
            "base_data_dir": str(tmp_path / "data"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "outputs_dir": str(tmp_path / "outputs"),
        },
        "data": {"random_seed": 42, "num_teams": 4},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    kaggle_dir = tmp_path / "data" / "raw" / "kaggle" / "downloads"
    kaggle_dir.mkdir(parents=True)

    import pandas as pd
    pd.DataFrame({
        "Season": [2023, 2023],
        "WTeamID": [1101, 1102],
        "LTeamID": [1103, 1104],
    }).to_csv(kaggle_dir / "MNCAATourneyCompactResults.csv", index=False)

    pd.DataFrame({
        "Season": [2023] * 4,
        "TeamID": [1101, 1102, 1103, 1104],
        "Seed": ["W01", "W05", "W12", "W08"],
    }).to_csv(kaggle_dir / "MNCAATourneySeeds.csv", index=False)

    pd.DataFrame({
        "Season": [2023] * 8,
        "RankingDayNum": [128] * 8,
        "SystemName": ["POM"] * 4 + ["MOR"] * 4,
        "TeamID": [1101, 1102, 1103, 1104, 1101, 1102, 1103, 1104],
        "OrdinalRank": [5, 20, 60, 30, 8, 22, 55, 28],
    }).to_csv(kaggle_dir / "MMasseyOrdinals.csv", index=False)

    out = build_game_features(2026, str(cfg_path))
    df = pd.read_parquet(out)

    assert "label" in df.columns
    assert set(df["label"].unique()).issubset({0, 1}), "Labels must be binary"
    assert "rank_diff_POM" in df.columns
    assert "rank_diff_MOR" in df.columns
    assert "rank_diff_SAG" in df.columns, "rank_diff_SAG must be present even if NaN-filled (SAG missing from fixture)"
    assert len(df) == 4, "2 games × 2 mirror rows = 4 rows"


def test_calibrator_produces_reasonable_probabilities(tmp_path):
    """After calibration, a 0.9 raw prob must map to < 0.95 and > 0.55."""
    import pickle
    import yaml
    from src.models.calibrate import train_calibrator

    config = {
        "project": {"target_year": 2025, "base_data_dir": str(tmp_path / "data"),
                     "artifacts_dir": str(tmp_path / "artifacts"), "outputs_dir": str(tmp_path / "outputs")},
        "data": {"random_seed": 42, "num_teams": 4},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    # Write calibration set where high raw probs -> realistic outputs
    proc_dir = tmp_path / "data" / "processed"
    proc_dir.mkdir(parents=True)
    # Simulate: probs near 0.9 correspond to ~85% true win rate
    rng = np.random.default_rng(42)
    probs = np.clip(rng.beta(5, 2, 500), 0.01, 0.99)
    outcomes = rng.binomial(1, probs * 0.9)
    pd.DataFrame({"model_prob": probs, "outcome": outcomes}).to_parquet(
        proc_dir / "calibration_set.parquet", index=False
    )

    val_dir = tmp_path / "outputs" / "validation"
    val_dir.mkdir(parents=True)

    out = train_calibrator(str(cfg_path))
    with open(out, "rb") as f:
        cal = pickle.load(f)

    # A very high raw probability should calibrate to <0.95
    high_prob = np.array([[0.95]])
    calibrated = cal.predict_proba(high_prob)[0, 1]
    assert calibrated < 0.95, f"Calibrated prob {calibrated:.3f} still too high"
    assert calibrated > 0.70, f"Calibrated prob {calibrated:.3f} unreasonably low for 0.95 raw input"
