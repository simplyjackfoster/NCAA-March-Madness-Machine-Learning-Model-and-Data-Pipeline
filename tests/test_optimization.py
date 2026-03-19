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
