from __future__ import annotations

import pandas as pd

# Historical ESPN Tournament Challenge championship pick rates per team by seed.
# Source: ESPN Tournament Challenge 2013-2024 averages.
# These are per-team rates (e.g., each of the four 1-seeds gets ~14.5%).
_SEED_PICK_RATES: dict[int, float] = {
    1: 0.145,   # ~14.5% each → 58% total for 4 one-seeds (spec: ~58%)
    2: 0.062,   # ~6.2% each  → 25% total for 4 two-seeds  (spec: ~25%)
    3: 0.025,   # ~2.5% each  → 10% total                  (spec: ~10%)
    4: 0.006,   # ~0.6% each                                (spec: ~0.6%)
    5: 0.006,   # ~0.6% each
    6: 0.002,   # ~0.2% each                                (spec: ~0.2%)
    7: 0.002,
    8: 0.001,
    9: 0.001,
    10: 0.001,
    11: 0.001,
    12: 0.0005,
    13: 0.0003,
    14: 0.0002,
    15: 0.0001,
    16: 0.0001,
}

# Extra pick share for historically popular programs.
# These are added on top of the seed base rate before normalization.
_DEFAULT_NAME_BOOSTS: dict[str, float] = {
    "Duke": 0.015,
    "Kentucky": 0.015,
    "Kansas": 0.010,
    "North Carolina": 0.010,
    "Michigan": 0.005,
    "UCLA": 0.005,
    "Florida": 0.005,
    "Arizona": 0.005,
    "Michigan St.": 0.003,
    "Connecticut": 0.003,
}


def get_seed_popularity(
    bracket_df: pd.DataFrame,
    name_boosts: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return championship pick probability for each team based on seed and name recognition.

    Args:
        bracket_df: DataFrame with columns ``team_name`` and ``seed``.
        name_boosts: Optional override for name-recognition boosts.
                     Defaults to ``_DEFAULT_NAME_BOOSTS``.

    Returns:
        Dict mapping team_name -> pick_pct, normalized to sum to 1.0.
    """
    boosts = _DEFAULT_NAME_BOOSTS if name_boosts is None else name_boosts
    picks: dict[str, float] = {}
    for _, row in bracket_df.iterrows():
        team = str(row["team_name"])
        seed = int(row["seed"])
        base = _SEED_PICK_RATES.get(seed, 0.0001)
        picks[team] = base + boosts.get(team, 0.0)

    total = sum(picks.values()) or 1.0
    return {t: v / total for t, v in picks.items()}
