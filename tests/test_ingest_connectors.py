from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.build_crosswalk import build_crosswalk
from src.data.ingest_barttorvik import ingest_barttorvik
from src.data.ingest_bracket import ingest_bracket
from src.data.ingest_kaggle import ingest_kaggle
from src.data.ingest_kenpom import ingest_kenpom


def _write_config(config_path: Path, data: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_ingest_bracket_from_local_csv(tmp_path: Path):
    bracket_csv = tmp_path / "bracket.csv"
    pd.DataFrame(
        [
            {"team_name": "Team_01", "seed": 1, "region": "East", "slot": 1},
            {"team_name": "Team_02", "seed": 16, "region": "East", "slot": 2},
        ]
    ).to_csv(bracket_csv, index=False)

    cfg = {
        "project": {"target_year": 2026},
        "data": {"random_seed": 42, "num_teams": 64},
        "data_sources": {"bracket": {"local_path": str(bracket_csv)}},
        "simulation": {"num_sims": 10},
        "optimization": {"min_champion_prob": 0.02, "risk_tolerance": "balanced", "pool_size": 50},
    }
    cfg_path = tmp_path / "test_ingest_bracket.yaml"
    _write_config(cfg_path, cfg)

    out_path = ingest_bracket(2026, str(cfg_path))
    out_df = pd.read_csv(out_path)

    assert len(out_df) == 2
    assert set(out_df.columns) == {"season", "team_name", "seed", "region", "slot"}


def test_ingest_kaggle_normalizes_compact_columns(tmp_path: Path):
    games_csv = tmp_path / "games.csv"
    pd.DataFrame(
        [
            {"Season": 2026, "WTeamID": 1, "LTeamID": 2, "WScore": 72, "LScore": 66, "DayNum": 10},
            {"Season": 2026, "WTeamID": 2, "LTeamID": 1, "WScore": 70, "LScore": 67, "DayNum": 12},
        ]
    ).to_csv(games_csv, index=False)

    cfg = {
        "project": {"target_year": 2026},
        "data": {"random_seed": 42, "num_teams": 64},
        "data_sources": {"kaggle": {"local_path": str(games_csv)}},
        "simulation": {"num_sims": 10},
        "optimization": {"min_champion_prob": 0.02, "risk_tolerance": "balanced", "pool_size": 50},
    }
    cfg_path = tmp_path / "test_ingest_kaggle.yaml"
    _write_config(cfg_path, cfg)

    out_path = ingest_kaggle(2026, str(cfg_path))
    out_df = pd.read_csv(out_path)

    assert list(out_df.columns) == ["season", "game_id", "team_id", "opp_id", "score", "opp_score", "won"]
    assert out_df["won"].tolist() == [1, 1]


def test_ingest_barttorvik_from_local_csv(tmp_path: Path):
    """Barttorvik ingest reads local CSV and normalizes to expected columns."""
    bt_csv = tmp_path / "barttorvik.csv"
    pd.DataFrame([
        {"team_name": "Duke", "adj_o": 120.5, "adj_d": 88.3, "tempo": 71.2},
        {"team_name": "Kansas", "adj_o": 118.1, "adj_d": 91.0, "tempo": 68.5},
    ]).to_csv(bt_csv, index=False)

    cfg = {
        "project": {"target_year": 2026},
        "data": {"random_seed": 42, "num_teams": 64},
        "data_sources": {"barttorvik": {"local_path": str(bt_csv)}},
        "simulation": {"num_sims": 10},
        "optimization": {"min_champion_prob": 0.02, "risk_tolerance": "balanced", "pool_size": 50},
    }
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, cfg)

    out_path = ingest_barttorvik(2026, str(cfg_path))
    out_df = pd.read_csv(out_path)

    assert len(out_df) == 2
    assert "team_name" in out_df.columns
    assert "adj_o" in out_df.columns
    assert "adj_d" in out_df.columns
    assert "tempo" in out_df.columns
    assert out_df.loc[out_df["team_name"] == "Duke", "adj_o"].iloc[0] == pytest.approx(120.5)


def test_ingest_kenpom_from_local_csv(tmp_path: Path):
    """KenPom ingest reads local CSV with required columns and normalizes correctly."""
    kp_csv = tmp_path / "kenpom.csv"
    pd.DataFrame([
        {"team_name": "Duke", "adj_em": 32.1, "adj_t": 70.5, "luck": 0.021},
        {"team_name": "Kansas", "adj_em": 28.4, "adj_t": 68.2, "luck": -0.011},
    ]).to_csv(kp_csv, index=False)

    cfg = {
        "project": {"target_year": 2026},
        "data": {"random_seed": 42, "num_teams": 64},
        "data_sources": {"kenpom": {"local_path": str(kp_csv)}},
        "simulation": {"num_sims": 10},
        "optimization": {"min_champion_prob": 0.02, "risk_tolerance": "balanced", "pool_size": 50},
    }
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, cfg)

    out_path = ingest_kenpom(2026, str(cfg_path))
    out_df = pd.read_csv(out_path)

    assert len(out_df) == 2
    assert set(out_df.columns) == {"season", "team_name", "adj_em", "adj_t", "luck"}
    assert out_df.loc[out_df["team_name"] == "Duke", "adj_em"].iloc[0] == pytest.approx(32.1)


def test_build_crosswalk_uses_real_kaggle_team_ids(tmp_path: Path):
    """When MTeams.csv is present, crosswalk assigns real Kaggle TeamIDs."""
    bt_csv = tmp_path / "barttorvik_2026.csv"
    pd.DataFrame([
        {"team_name": "Duke", "adj_o": 120.0, "adj_d": 90.0, "tempo": 70.0},
        {"team_name": "Kansas", "adj_o": 118.0, "adj_d": 92.0, "tempo": 68.0},
    ]).to_csv(bt_csv, index=False)

    mteams_path = tmp_path / "MTeams.csv"
    pd.DataFrame([
        {"TeamID": 1181, "TeamName": "Duke"},
        {"TeamID": 1242, "TeamName": "Kansas"},
        {"TeamID": 1999, "TeamName": "Some Other Team"},
    ]).to_csv(mteams_path, index=False)

    cfg = {
        "project": {"target_year": 2026},
        "data": {"random_seed": 42, "num_teams": 64},
        "data_sources": {
            "barttorvik": {"local_path": str(bt_csv)},
            "kenpom": {"local_path": None},
            "kaggle": {"local_path": None},
            "bracket": {"local_path": None},
        },
        "crosswalk": {"mteams_path": str(mteams_path)},
        "simulation": {"num_sims": 10},
        "optimization": {"min_champion_prob": 0.02, "risk_tolerance": "balanced", "pool_size": 50},
    }
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, cfg)

    out_path = build_crosswalk(2026, str(cfg_path))
    cw = pd.read_csv(out_path)

    duke_id = cw.loc[cw["barttorvik_name"] == "Duke", "kaggle_team_id"].iloc[0]
    kansas_id = cw.loc[cw["barttorvik_name"] == "Kansas", "kaggle_team_id"].iloc[0]
    assert duke_id == 1181
    assert kansas_id == 1242


def test_build_crosswalk_falls_back_to_sequential_without_mteams(tmp_path: Path):
    """When no mteams_path is configured, crosswalk falls back to sequential IDs."""
    bt_csv = tmp_path / "barttorvik_2026.csv"
    pd.DataFrame([
        {"team_name": "Duke", "adj_o": 120.0, "adj_d": 90.0, "tempo": 70.0},
    ]).to_csv(bt_csv, index=False)

    cfg = {
        "project": {"target_year": 2026},
        "data": {"random_seed": 42, "num_teams": 64},
        "data_sources": {
            "barttorvik": {"local_path": str(bt_csv)},
            "kenpom": {"local_path": None},
            "kaggle": {"local_path": None},
            "bracket": {"local_path": None},
        },
        "simulation": {"num_sims": 10},
        "optimization": {"min_champion_prob": 0.02, "risk_tolerance": "balanced", "pool_size": 50},
    }
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, cfg)

    out_path = build_crosswalk(2026, str(cfg_path))
    cw = pd.read_csv(out_path)

    assert len(cw) == 1
    assert cw.iloc[0]["kaggle_team_id"] == 1  # sequential fallback
