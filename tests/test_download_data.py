from pathlib import Path
import pandas as pd
import pytest
from scripts.download_data import process_game_results, process_teams


def test_process_game_results_filters_by_single_season(tmp_path):
    """Only rows matching the target season are kept."""
    csv = tmp_path / "results.csv"
    pd.DataFrame([
        {"Season": 2025, "WTeamID": 1101, "LTeamID": 1102, "WScore": 70, "LScore": 60, "DayNum": 10},
        {"Season": 2026, "WTeamID": 1103, "LTeamID": 1104, "WScore": 75, "LScore": 68, "DayNum": 12},
        {"Season": 2026, "WTeamID": 1104, "LTeamID": 1103, "WScore": 80, "LScore": 71, "DayNum": 14},
    ]).to_csv(csv, index=False)

    result = process_game_results(csv, season=2026)

    assert len(result) == 2
    assert all(result["Season"] == 2026)


def test_process_game_results_filters_by_multiple_seasons(tmp_path):
    """Passing a list of seasons keeps all matching rows."""
    csv = tmp_path / "results.csv"
    pd.DataFrame([
        {"Season": 2024, "WTeamID": 1, "LTeamID": 2, "WScore": 70, "LScore": 60, "DayNum": 1},
        {"Season": 2025, "WTeamID": 3, "LTeamID": 4, "WScore": 72, "LScore": 65, "DayNum": 2},
        {"Season": 2026, "WTeamID": 5, "LTeamID": 6, "WScore": 68, "LScore": 62, "DayNum": 3},
        {"Season": 2027, "WTeamID": 7, "LTeamID": 8, "WScore": 81, "LScore": 74, "DayNum": 4},
    ]).to_csv(csv, index=False)

    result = process_game_results(csv, season=[2024, 2025, 2026])

    assert len(result) == 3
    assert set(result["Season"].unique()) == {2024, 2025, 2026}


def test_process_teams_returns_required_columns(tmp_path):
    """process_teams returns a DataFrame with TeamID and TeamName columns."""
    csv = tmp_path / "MTeams.csv"
    pd.DataFrame([
        {"TeamID": 1181, "TeamName": "Duke", "FirstD1Season": 1985, "LastD1Season": 2026},
        {"TeamID": 1242, "TeamName": "Kansas", "FirstD1Season": 1985, "LastD1Season": 2026},
    ]).to_csv(csv, index=False)

    result = process_teams(csv)

    assert "TeamID" in result.columns
    assert "TeamName" in result.columns
    assert len(result) == 2
    # Extra columns should not be present
    assert list(result.columns) == ["TeamID", "TeamName"]
