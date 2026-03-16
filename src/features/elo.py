from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EloConfig:
    k_factor: float = 20.0
    initial_rating: float = 1500.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_winner: float, rating_loser: float, cfg: EloConfig | None = None) -> tuple[float, float]:
    cfg = cfg or EloConfig()
    p_win = expected_score(rating_winner, rating_loser)
    delta = cfg.k_factor * (1 - p_win)
    return rating_winner + delta, rating_loser - delta
