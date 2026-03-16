from src.simulation.tournament_sim import run_simulation
from src.simulation.score_sim import simulate_scores

if __name__ == "__main__":
    run_simulation(2026)
    simulate_scores(2026)
    print("Simulation complete")
