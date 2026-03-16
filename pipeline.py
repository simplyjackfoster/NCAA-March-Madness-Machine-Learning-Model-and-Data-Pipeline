from __future__ import annotations

import argparse

from src.data.ingest_barttorvik import ingest_barttorvik
from src.data.build_crosswalk import build_crosswalk
from src.features.team_features import build_team_features
from src.features.game_features import build_game_features
from src.models.prior_model import train_prior_model
from src.models.loyo_validator import run_loyo
from src.models.lgbm_model import train_lgbm_proxy
from src.models.calibrate import train_calibrator
from src.models.xgb_model import train_xgb_model
from src.simulation.matchup_matrix import build_matchup_matrix
from src.simulation.tournament_sim import run_simulation
from src.field.pool_model import build_pool_model
from src.optimization.champion_search import run_champion_search
from src.optimization.greedy_optimizer import select_final_bracket
from src.export.bracket_formatter import export_bracket_text


def run_pipeline(year: int, config: str = "configs/config.yaml"):
    ingest_barttorvik(year, config)
    build_crosswalk(year, config)
    build_team_features(year, config)
    build_game_features(year, config)
    train_prior_model(config)
    run_loyo(config)
    train_lgbm_proxy(config)
    train_xgb_model(config)
    train_calibrator(config)
    build_matchup_matrix(year, config)
    run_simulation(year, config)
    build_pool_model(year, config)
    run_champion_search(year, config)
    select_final_bracket(year, config)
    export_bracket_text(year, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run March Madness office pool pipeline")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_pipeline(args.year, args.config)
    print("Pipeline complete.")
