from src.optimization.champion_search import run_champion_search
from src.optimization.expected_score import compute_expected_scores
from src.optimization.leverage import compute_leverage
from src.optimization.greedy_optimizer import select_final_bracket
from src.optimization.annealing import run_annealing

if __name__ == "__main__":
    year = 2026
    run_champion_search(year)
    compute_expected_scores(year)
    compute_leverage(year)
    select_final_bracket(year)
    run_annealing(year)
    print("Optimization complete")
