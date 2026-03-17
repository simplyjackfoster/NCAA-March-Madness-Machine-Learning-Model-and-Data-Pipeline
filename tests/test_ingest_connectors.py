from pathlib import Path

import pandas as pd
import yaml

from src.data.ingest_bracket import ingest_bracket
from src.data.ingest_kaggle import ingest_kaggle


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
