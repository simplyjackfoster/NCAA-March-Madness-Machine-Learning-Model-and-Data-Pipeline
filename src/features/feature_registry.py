from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    stage: str
    description: str


FEATURE_REGISTRY: list[FeatureSpec] = [
    FeatureSpec("adj_o", "team", "Adjusted offensive efficiency"),
    FeatureSpec("adj_d", "team", "Adjusted defensive efficiency"),
    FeatureSpec("tempo", "team", "Adjusted possessions per game"),
    FeatureSpec("net_rating", "team", "adj_o - adj_d"),
    FeatureSpec("elo_pre", "team", "Pre-tournament Elo"),
    FeatureSpec("seed_diff", "game", "Seed difference team - opponent"),
    FeatureSpec("rank_diff_POM", "game", "KenPom (POM) Massey ordinal rank difference"),
    FeatureSpec("rank_diff_MOR", "game", "Massey (MOR) ordinal rank difference"),
    FeatureSpec("rank_diff_SAG", "game", "Sagarin (SAG) ordinal rank difference"),
]


def get_feature_names(stage: str | None = None) -> list[str]:
    if stage is None:
        return [f.name for f in FEATURE_REGISTRY]
    return [f.name for f in FEATURE_REGISTRY if f.stage == stage]
