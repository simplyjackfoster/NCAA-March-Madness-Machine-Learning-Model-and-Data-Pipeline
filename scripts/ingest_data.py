from src.data.ingest_barttorvik import ingest_barttorvik
from src.data.ingest_kaggle import ingest_kaggle
from src.data.ingest_kenpom import ingest_kenpom
from src.data.build_crosswalk import build_crosswalk
from src.data.ingest_bracket import ingest_bracket

if __name__ == "__main__":
    year = 2026
    ingest_barttorvik(year)
    ingest_kaggle(year)
    ingest_kenpom(year)
    ingest_bracket(year)
    build_crosswalk(year)
    print("Ingestion complete")
