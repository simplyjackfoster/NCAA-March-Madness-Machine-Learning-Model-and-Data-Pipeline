from src.features.team_features import build_team_features
from src.features.game_features import build_game_features

if __name__ == "__main__":
    year = 2026
    build_team_features(year)
    build_game_features(year)
    print("Feature building complete")
