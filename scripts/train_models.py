from src.models.prior_model import train_prior_model
from src.models.lgbm_model import train_lgbm_proxy
from src.models.xgb_model import train_xgb_model
from src.models.ensemble import build_ensemble_weights
from src.models.loyo_validator import run_loyo

if __name__ == "__main__":
    train_prior_model()
    train_lgbm_proxy()
    train_xgb_model()
    build_ensemble_weights()
    run_loyo()
    print("Training complete")
