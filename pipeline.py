from __future__ import annotations

import argparse

from src.data.ingest_barttorvik import ingest_barttorvik
from src.data.ingest_kaggle import ingest_kaggle
from src.data.ingest_kenpom import ingest_kenpom
from src.data.build_crosswalk import build_crosswalk
from src.data.ingest_bracket import ingest_bracket
from src.features.team_features import build_team_features
from src.features.game_features import build_game_features
from src.models.prior_model import train_prior_model
from src.models.loyo_validator import run_loyo
from src.models.lgbm_model import train_lgbm_proxy
from src.models.calibrate import train_calibrator
from src.models.xgb_model import train_xgb_model
from src.models.ensemble import build_ensemble_weights
from src.simulation.matchup_matrix import build_matchup_matrix
from src.simulation.tournament_sim import run_simulation
from src.simulation.score_sim import simulate_scores
from src.field.pool_model import build_pool_model
from src.field.field_sampler import sample_field_brackets
from src.optimization.champion_search import run_champion_search
from src.optimization.expected_score import compute_expected_scores
from src.optimization.leverage import compute_leverage
from src.optimization.greedy_optimizer import select_final_bracket
from src.optimization.annealing import run_annealing
from src.export.bracket_formatter import export_bracket_text
from src.export.strategy_report import export_strategy_report


def run_pipeline(year: int, config: str = "configs/config.yaml"):
    ingest_barttorvik(year, config)
    ingest_kaggle(year, config)
    ingest_kenpom(year, config)
    ingest_bracket(year, config)
    build_crosswalk(year, config)
    build_team_features(year, config)
    build_game_features(year, config)
    train_prior_model(config)
    run_loyo(config)
    train_lgbm_proxy(config)
    train_xgb_model(config)
    build_ensemble_weights(config)
    train_calibrator(config)
    build_matchup_matrix(year, config)
    run_simulation(year, config)
    simulate_scores(year, config)
    build_pool_model(year, config)
    sample_field_brackets(year, config)
    run_champion_search(year, config)
    compute_expected_scores(year, config)
    compute_leverage(year, config)
    select_final_bracket(year, config)
    run_annealing(year, config)
    export_bracket_text(year, config)
    export_strategy_report(year, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run March Madness office pool pipeline")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_pipeline(args.year, args.config)
    print("Pipeline complete.")
