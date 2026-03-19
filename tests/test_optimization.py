import pandas as pd


def test_leverage_computation_shape():
    df = pd.DataFrame({"champ_prob": [0.2], "field_pick": [0.1]})
    df["lev"] = df["champ_prob"] / df["field_pick"]
    assert df["lev"].iloc[0] == 2.0


def test_seed_popularity_sums_to_one():
    """Pick rates must sum to 1.0 across all 64 bracket teams."""
    from src.field.seed_popularity import get_seed_popularity

    bracket = pd.DataFrame({
        "team_name": [f"Team{i}" for i in range(64)],
        "seed": ([1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15] * 4),
    })
    picks = get_seed_popularity(bracket)
    assert abs(sum(picks.values()) - 1.0) < 1e-6, "pick rates must sum to 1.0"
    assert all(v > 0 for v in picks.values()), "every team must have non-zero pick rate"


def test_seed_popularity_one_seeds_dominate():
    """1-seeds must collectively have higher pick rate than all other seeds combined."""
    from src.field.seed_popularity import get_seed_popularity

    bracket = pd.DataFrame({
        "team_name": [f"Team{i}" for i in range(64)],
        "seed": ([1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15] * 4),
    })
    picks = get_seed_popularity(bracket)
    one_seed_teams = bracket.loc[bracket["seed"] == 1, "team_name"].tolist()
    one_seed_total = sum(picks[t] for t in one_seed_teams)
    assert one_seed_total > 0.50, f"1-seeds should dominate with >50%; got {one_seed_total:.2f}"


def test_seed_popularity_name_boost_applied():
    """Duke should have higher pick rate than a generic seed-1 team."""
    from src.field.seed_popularity import get_seed_popularity

    bracket = pd.DataFrame({
        "team_name": ["Duke", "Generic1", "Generic2", "Generic3"] + [f"Team{i}" for i in range(60)],
        "seed": [1, 1, 1, 1] + ([16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15] * 4),
    })
    picks = get_seed_popularity(bracket)
    assert picks["Duke"] > picks["Generic1"], "Duke should get a name-recognition boost"


def test_espn_loader_uses_seed_model(tmp_path):
    """espn_picks JSON must reflect seed-based rates, not circular model output."""
    import json
    import yaml
    from src.field.espn_loader import load_espn_pick_rates

    config = {
        "project": {"target_year": 2025, "base_data_dir": str(tmp_path / "data"),
                     "artifacts_dir": str(tmp_path / "artifacts"), "outputs_dir": str(tmp_path / "outputs")},
        "data": {"random_seed": 42, "num_teams": 4},
        "data_sources": {},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    year = 2025
    # Write bracket with seeds
    br_dir = tmp_path / "data" / "raw" / "bracket"
    br_dir.mkdir(parents=True)
    pd.DataFrame({
        "team_name": ["TeamA", "TeamB", "TeamC", "TeamD"],
        "seed": [1, 16, 8, 9],
        "region": ["East"] * 4,
        "slot": [1, 2, 3, 4],
    }).to_csv(br_dir / f"bracket_{year}.csv", index=False)

    field_dir = tmp_path / "data" / "field"
    field_dir.mkdir(parents=True)

    out = load_espn_pick_rates(year, str(cfg_path))
    picks = json.loads(out.read_text())

    # Seed-1 team (TeamA) must have much higher pick rate than seed-16 (TeamB)
    assert picks["TeamA"] > picks["TeamB"] * 10, (
        f"Seed-1 TeamA ({picks['TeamA']:.4f}) should dominate seed-16 TeamB ({picks['TeamB']:.4f})"
    )
    assert abs(sum(picks.values()) - 1.0) < 1e-6


def test_pool_simulator_score_bracket():
    """_score_bracket must return correct ESPN points for a 2-team bracket."""
    import numpy as np
    from src.simulation.pool_simulator import _build_bracket, _score_bracket

    mat = np.array([[0.0, 0.7], [0.3, 0.0]])
    picks = _build_bracket([0, 1], mat, rng=None, forced_champion=0)
    assert _score_bracket(picks, {1: [0]}) == 32, "Championship correct pick = 32 pts"
    assert _score_bracket(picks, {1: [1]}) == 0, "Championship wrong pick = 0 pts"


def test_pool_simulator_analytical_p_win(tmp_path):
    """With pool_size=1 opponent always picking the weaker team, P(win pool) ≈ mat[0,1]."""
    import json, yaml
    import numpy as np
    import pandas as pd
    from src.simulation.pool_simulator import run_pool_simulation

    # 2-team bracket: team 0 beats team 1 with prob 0.7
    # User picks team 0; single opponent always picks team 1.
    # P(user wins pool) = P(team 0 wins tournament) ≈ 0.7
    config = {
        "project": {"target_year": 2025, "base_data_dir": str(tmp_path / "data"),
                     "artifacts_dir": str(tmp_path / "artifacts"), "outputs_dir": str(tmp_path / "outputs")},
        "data": {"random_seed": 0, "num_teams": 2},
        "simulation": {"num_sims": 5000},
        "optimization": {"pool_size": 1, "pool_sims": 5000, "min_champion_prob": 0.0,
                         "risk_tolerance": "balanced"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    year = 2025
    br_dir = tmp_path / "data" / "raw" / "bracket"
    br_dir.mkdir(parents=True)
    pd.DataFrame({
        "team_name": ["Strong", "Weak"], "seed": [1, 16],
        "region": ["East", "East"], "slot": [1, 2],
    }).to_csv(br_dir / f"bracket_{year}.csv", index=False)

    feat_dir = tmp_path / "data" / "features"
    feat_dir.mkdir(parents=True)
    pd.DataFrame({
        "season": [year, year], "kaggle_team_id": [1, 2],
        "display_name": ["Strong", "Weak"],
        "adj_o": [115.0, 95.0], "adj_d": [85.0, 105.0], "tempo": [68.0, 68.0],
        "net_rating": [30.0, -10.0], "elo_pre": [1740.0, 1340.0],
        "adj_em": [20.0, -10.0], "luck": [0.0, 0.0], "seed": [1, 16],
    }).to_parquet(feat_dir / f"team_season_{year}.parquet", index=False)

    mat = np.array([[0.0, 0.7], [0.3, 0.0]])
    tourney_dir = tmp_path / "data" / "tournament"
    tourney_dir.mkdir(parents=True)
    np.save(tourney_dir / f"prob_matrix_{year}.npy", mat)

    sim_dir = tmp_path / "data" / "simulation"
    sim_dir.mkdir(parents=True)
    (sim_dir / f"advance_probs_{year}.json").write_text(json.dumps(
        {"champion_prob": {"Strong": 0.70, "Weak": 0.30}}
    ))
    field_dir = tmp_path / "data" / "field"
    field_dir.mkdir(parents=True)
    # Opponent always picks Weak (seed 16): pick_pct = 1.0
    (field_dir / f"espn_picks_{year}.json").write_text(json.dumps(
        {"Strong": 0.0, "Weak": 1.0}
    ))

    out = run_pool_simulation(year, str(cfg_path))
    df = pd.read_parquet(out)
    strong_row = df[df["champion"] == "Strong"].iloc[0]

    # P(user wins) when picking Strong ≈ P(Strong wins tournament) ≈ 0.7
    assert 0.55 < strong_row["p_win_pool"] < 0.85, (
        f"Expected P(win pool) ≈ 0.7 for Strong, got {strong_row['p_win_pool']:.3f}"
    )


def test_pool_simulator_p_win_in_range(tmp_path):
    """P(win pool) must be in [0, 1] for all candidates."""
    import yaml
    import numpy as np
    import pandas as pd
    from src.simulation.pool_simulator import run_pool_simulation

    config = {
        "project": {"target_year": 2025, "base_data_dir": str(tmp_path / "data"),
                     "artifacts_dir": str(tmp_path / "artifacts"), "outputs_dir": str(tmp_path / "outputs")},
        "data": {"random_seed": 42, "num_teams": 4},
        "simulation": {"num_sims": 5000},
        "optimization": {"pool_size": 5, "pool_sims": 200, "min_champion_prob": 0.02,
                         "risk_tolerance": "balanced"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    year = 2025
    # Write 4-team bracket
    br_dir = tmp_path / "data" / "raw" / "bracket"
    br_dir.mkdir(parents=True)
    pd.DataFrame({
        "team_name": ["TeamA", "TeamB", "TeamC", "TeamD"],
        "seed": [1, 16, 2, 15], "region": ["East"] * 4, "slot": [1, 2, 3, 4],
    }).to_csv(br_dir / f"bracket_{year}.csv", index=False)

    # Write team features
    feat_dir = tmp_path / "data" / "features"
    feat_dir.mkdir(parents=True)
    pd.DataFrame({
        "season": [year] * 4, "kaggle_team_id": [1, 2, 3, 4],
        "display_name": ["TeamA", "TeamB", "TeamC", "TeamD"],
        "adj_o": [110.0, 95.0, 105.0, 90.0], "adj_d": [90.0, 105.0, 95.0, 110.0],
        "tempo": [70.0] * 4, "net_rating": [20.0, -10.0, 10.0, -20.0],
        "elo_pre": [1660.0, 1420.0, 1580.0, 1340.0],
        "adj_em": [20.0, -10.0, 10.0, -20.0], "luck": [0.0] * 4, "seed": [1, 16, 2, 15],
    }).to_parquet(feat_dir / f"team_season_{year}.parquet", index=False)

    # Write prob matrix (4x4)
    mat = np.array([
        [0.00, 0.90, 0.60, 0.95],
        [0.10, 0.00, 0.15, 0.80],
        [0.40, 0.85, 0.00, 0.92],
        [0.05, 0.20, 0.08, 0.00],
    ])
    tourney_dir = tmp_path / "data" / "tournament"
    tourney_dir.mkdir(parents=True)
    np.save(tourney_dir / f"prob_matrix_{year}.npy", mat)

    # Write advance_probs (from tournament sim)
    sim_dir = tmp_path / "data" / "simulation"
    sim_dir.mkdir(parents=True)
    import json
    (sim_dir / f"advance_probs_{year}.json").write_text(json.dumps({
        "champion_prob": {"TeamA": 0.55, "TeamB": 0.05, "TeamC": 0.35, "TeamD": 0.05}
    }))

    # Write espn_picks (from espn_loader)
    field_dir = tmp_path / "data" / "field"
    field_dir.mkdir(parents=True)
    (field_dir / f"espn_picks_{year}.json").write_text(json.dumps({
        "TeamA": 0.50, "TeamB": 0.02, "TeamC": 0.40, "TeamD": 0.08
    }))

    out = run_pool_simulation(year, str(cfg_path))
    df = pd.read_parquet(out)

    assert "p_win_pool" in df.columns
    assert df["p_win_pool"].between(0, 1).all()
    assert len(df) == 4  # one row per team
    # Top P(win pool) should be for either TeamA or TeamC (both strong)
    assert df.iloc[0]["champion"] in ("TeamA", "TeamC")


def test_leverage_preserves_p_win_pool(tmp_path):
    """compute_leverage must not overwrite p_win_pool from pool simulator."""
    import yaml
    import pandas as pd
    from src.optimization.leverage import compute_leverage

    config = {
        "project": {"target_year": 2025, "base_data_dir": str(tmp_path / "data"),
                     "artifacts_dir": str(tmp_path / "artifacts"), "outputs_dir": str(tmp_path / "outputs")},
        "data": {"random_seed": 42, "num_teams": 4},
        "optimization": {"pool_size": 20, "pool_sims": 200, "min_champion_prob": 0.02,
                         "risk_tolerance": "balanced"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))

    opt_dir = tmp_path / "data" / "optimization"
    opt_dir.mkdir(parents=True)
    # Write expected_scores with a known p_win_pool value
    pd.DataFrame({
        "champion": ["TeamA", "TeamB"],
        "champ_prob": [0.35, 0.20],
        "field_pick": [0.15, 0.08],
        "leverage_score": [2.33, 2.50],
        "expected_score": [67.2, 38.4],
        "p_win_pool": [0.124, 0.091],   # from pool simulator — must be preserved
    }).to_parquet(opt_dir / "expected_scores_2025.parquet", index=False)

    out = compute_leverage(2025, str(cfg_path))
    df = pd.read_parquet(out)

    assert df.loc[df["champion"] == "TeamA", "p_win_pool"].iloc[0] == 0.124, \
        "p_win_pool must be preserved from pool simulator"
    assert df.loc[df["champion"] == "TeamB", "p_win_pool"].iloc[0] == 0.091
