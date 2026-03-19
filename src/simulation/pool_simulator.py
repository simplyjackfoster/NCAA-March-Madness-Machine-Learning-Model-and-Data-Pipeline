from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.io import read_json, write_json

SCORING = [1, 2, 4, 8, 16, 32]  # ESPN points per round (rounds 1-6)


def _simulate_tournament(
    mat: np.ndarray, alive: list[int], rng: np.random.Generator
) -> dict[int, list[int]]:
    """Simulate one tournament outcome.

    Returns dict mapping round_num -> list of winner team indices.
    Winners from each round propagate forward as inputs to the next round.
    """
    outcomes: dict[int, list[int]] = {}
    current = list(alive)
    round_num = 0
    while len(current) > 1:
        round_num += 1
        next_round: list[int] = []
        winners: list[int] = []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            winner = a if rng.random() < mat[a, b] else b
            next_round.append(winner)
            winners.append(winner)
        outcomes[round_num] = winners
        current = next_round
    return outcomes


def _build_bracket(
    alive: list[int],
    mat: np.ndarray,
    rng: np.random.Generator | None,
    forced_champion: int | None = None,
) -> dict[int, list[int]]:
    """Build a full bracket pick set.

    If ``rng`` is None, picks greedily (always higher-probability team).
    If ``rng`` is provided, picks stochastically using prob_matrix.
    If ``forced_champion`` is set, forces that team to win every matchup they appear in.
    Winners from each round propagate forward sequentially.
    """
    current = list(alive)
    round_num = 0
    picks: dict[int, list[int]] = {}
    while len(current) > 1:
        round_num += 1
        next_round: list[int] = []
        picks[round_num] = []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            if forced_champion is not None and forced_champion in (a, b):
                winner = forced_champion
            elif rng is None:
                winner = a if mat[a, b] >= 0.5 else b
            else:
                winner = a if rng.random() < mat[a, b] else b
            next_round.append(winner)
            picks[round_num].append(winner)
        current = next_round
    return picks


def _score_bracket(
    picks: dict[int, list[int]],
    outcome: dict[int, list[int]],
    scoring: list[int] = SCORING,
) -> int:
    """Score a bracket against a tournament outcome using ESPN scoring.

    Rounds are scored from the end of the SCORING list so that the final
    round (championship) always maps to the highest point value regardless
    of how many teams are in the bracket.  For a standard 64-team bracket
    with 6 rounds, round 1 -> scoring[0]=1 pt and round 6 -> scoring[5]=32 pt.
    For a 2-team bracket with 1 round, that single round maps to scoring[5]=32 pt.
    """
    total = 0
    all_rounds = sorted(picks.keys())
    num_rounds = len(all_rounds)
    # Offset so the last round maps to scoring[-1] (championship = 32 pts)
    round_offset = len(scoring) - num_rounds
    for local_idx, round_num in enumerate(all_rounds):
        scoring_idx = round_offset + local_idx
        if scoring_idx < 0 or scoring_idx >= len(scoring):
            continue
        pts = scoring[scoring_idx]
        if round_num not in outcome:
            continue
        actual = set(outcome[round_num])
        for pick in picks[round_num]:
            if pick in actual:
                total += pts
    return total


def run_pool_simulation(year: int, config_path: str = "configs/config.yaml") -> Path:
    """Compute P(win pool) for each candidate champion via Monte Carlo simulation.

    Algorithm:
    1. Generate a fixed pool of ``pool_size`` opponent brackets (once, shared across
       all tournament simulations). Opponent champion picks are sampled from the
       seed-popularity field distribution; remaining picks filled sequentially via
       prob_matrix Bernoulli draws. Using a fixed pool isolates variance from the
       tournament outcome, which is the correct Monte Carlo design.
    2. Pre-simulate ``num_sims`` tournament outcomes.
    3. For each outcome, score all ``pool_size`` opponent brackets; record max score.
    4. For each candidate champion: build a greedy user bracket with that champion
       forced, score against each outcome, count wins vs. max opponent scores.

    Writes ``data/optimization/candidates_{year}.parquet`` with columns:
    champion, champ_prob, field_pick, leverage_score, expected_score, p_win_pool.

    Note: assumes len(ordered) is even at each round (standard bracket structure).
    """
    cfg = load_config(config_path)
    root = cfg["_root"]
    pool_size = int(cfg["optimization"]["pool_size"])
    num_sims = int(cfg["optimization"].get("pool_sims", 10000))
    rng = np.random.default_rng(cfg["data"]["random_seed"])

    mat = np.load(root / "data" / "tournament" / f"prob_matrix_{year}.npy")
    teams = pd.read_parquet(root / "data" / "features" / f"team_season_{year}.parquet")
    bracket = pd.read_csv(root / "data" / "raw" / "bracket" / f"bracket_{year}.csv")

    # Build ordered list of team indices (slots -> team positions in team_df)
    # Use only bracket columns needed to avoid duplicate 'seed' after merge
    bracket_slim = bracket[["team_name", "slot"]].copy()
    merged = bracket_slim.merge(
        teams.reset_index().rename(columns={"index": "team_idx"}),
        left_on="team_name", right_on="display_name", how="left",
    )
    ordered = (
        merged.sort_values("slot")["team_idx"]
        .dropna().astype(int).tolist()
    )

    # Load field pick distribution (team_name -> pct), convert to ordered weight array
    espn_picks = read_json(root / "data" / "field" / f"espn_picks_{year}.json")
    pick_weights = np.array([
        espn_picks.get(teams.iloc[t]["display_name"], 1e-6) for t in ordered
    ])
    pick_weights = pick_weights / pick_weights.sum()

    # Load champion probabilities from simulation
    adv = read_json(root / "data" / "simulation" / f"advance_probs_{year}.json")
    champ_probs = adv["champion_prob"]

    # Step 1: Generate a fixed pool of opponent brackets (generated once, not per-sim)
    # This is more efficient (pool_size brackets, not num_sims * pool_size) and
    # statistically correct (isolates tournament-outcome variance from bracket variance).
    fixed_opp_brackets = []
    for _ in range(pool_size):
        opp_champ_pos = int(rng.choice(len(ordered), p=pick_weights))
        opp_champ_idx = ordered[opp_champ_pos]
        fixed_opp_brackets.append(
            _build_bracket(ordered, mat, rng=rng, forced_champion=opp_champ_idx)
        )

    # Step 2: Pre-simulate all tournament outcomes
    all_outcomes = [_simulate_tournament(mat, ordered, rng) for _ in range(num_sims)]

    # Step 3: Pre-compute max opponent score per simulation
    max_opp_scores = [
        max(_score_bracket(opp, all_outcomes[i]) for opp in fixed_opp_brackets)
        for i in range(num_sims)
    ]

    # Step 4: Score each candidate champion
    # A user "wins the pool" when their score >= max opponent score (ties count as pool wins).
    rows = []
    for champ_team_idx in ordered:
        champ_name = teams.iloc[champ_team_idx]["display_name"]
        user_picks = _build_bracket(ordered, mat, rng=None, forced_champion=champ_team_idx)
        wins = sum(
            1
            for i in range(num_sims)
            if _score_bracket(user_picks, all_outcomes[i]) >= max_opp_scores[i]
        )
        model_p = champ_probs.get(champ_name, 0.0)
        field_p = espn_picks.get(champ_name, 1e-6)
        rows.append({
            "champion": champ_name,
            "champ_prob": model_p,
            "field_pick": field_p,
            "leverage_score": model_p / max(field_p, 1e-6),
            "expected_score": 192 * model_p,
            "p_win_pool": wins / num_sims,
        })

    df = pd.DataFrame(rows).sort_values("p_win_pool", ascending=False)
    out = root / "data" / "optimization" / f"candidates_{year}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(f"Wrote {run_pool_simulation(args.year, args.config)}")
