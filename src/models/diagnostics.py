from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.common.config import load_config
from src.common.io import write_json


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += (np.sum(mask) / len(y_true)) * abs(acc - conf)
    return float(ece)


def write_calibration_diagnostics(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> tuple[Path, Path]:
    metrics = {
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ece_10": expected_calibration_error(y_true, y_prob, n_bins=10),
    }

    bins = np.linspace(0.0, 1.0, 11)
    rows = []
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < 9 else y_prob <= hi)
        rows.append(
            {
                "bin": i,
                "bin_start": lo,
                "bin_end": hi,
                "count": int(np.sum(mask)),
                "avg_pred": float(np.mean(y_prob[mask])) if np.any(mask) else None,
                "empirical_win_rate": float(np.mean(y_true[mask])) if np.any(mask) else None,
            }
        )

    metrics_path = out_dir / "calibration_metrics.json"
    reliability_path = out_dir / "reliability_table.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_path, metrics)
    pd.DataFrame(rows).to_parquet(reliability_path, index=False)
    return metrics_path, reliability_path


def run_default_diagnostics(config_path: str = "configs/config.yaml") -> tuple[Path, Path]:
    cfg = load_config(config_path)
    root = cfg["_root"]
    train = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    probs = (train["elo_diff"].to_numpy() - train["elo_diff"].min())
    probs = probs / (probs.max() + 1e-9)
    y = train["label"].to_numpy()
    return write_calibration_diagnostics(y, probs, root / "outputs" / "validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    a, b = run_default_diagnostics(args.config)
    print(f"Wrote {a}\nWrote {b}")
