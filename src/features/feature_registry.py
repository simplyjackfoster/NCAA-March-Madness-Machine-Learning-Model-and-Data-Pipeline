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
    FeatureSpec("elo_diff", "game", "Elo difference team - opponent"),
    FeatureSpec("net_rating_diff", "game", "Net rating difference team - opponent"),
    FeatureSpec("tempo_diff", "game", "Tempo difference team - opponent"),
]


def get_feature_names(stage: str | None = None) -> list[str]:
    if stage is None:
        return [f.name for f in FEATURE_REGISTRY]
    return [f.name for f in FEATURE_REGISTRY if f.stage == stage]
