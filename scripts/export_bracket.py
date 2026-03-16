from src.export.bracket_formatter import export_bracket_text
from src.export.strategy_report import export_strategy_report

if __name__ == "__main__":
    year = 2026
    export_bracket_text(year)
    export_strategy_report(year)
    print("Export complete")
