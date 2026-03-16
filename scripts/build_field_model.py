from src.field.pool_model import build_pool_model
from src.field.field_sampler import sample_field_brackets

if __name__ == "__main__":
    build_pool_model(2026)
    sample_field_brackets(2026)
    print("Field model complete")
