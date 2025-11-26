#!/usr/bin/env python3
"""
03_train_baseline_model.py

Train simple baseline models to predict LN_IC50 from summary features.

Models:
- Ridge regression
- Random forest regressor

Outputs:
- results/models/baseline_metrics.csv
- results/models/{model_name}_ytrue_ypred.csv
- results/models/{model_name}_scatter.png
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger("cdr_ml.train")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    data_path: Path = PROJECT_ROOT / "results" / "processed" / "model_data.csv"
    out_dir: Path = PROJECT_ROOT / "results" / "models"

    target_column: str = "LN_IC50"
    feature_columns: tuple[str, ...] = ("COSMIC_ID", "DRUG_ID", "AUC", "Z_SCORE")

    test_size: float = 0.2
    random_state: int = 42


# ---------------------------------------------------------------------------
# ARGPARSE & LOGGING
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline models to predict drug response."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=TrainConfig().data_path,
        help="Path to processed model_data.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=TrainConfig().out_dir,
        help="Directory to write model outputs.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=TrainConfig().target_column,
        help="Target column name (default: LN_IC50).",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------------

def load_model_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    LOGGER.info("Loading processed data from %s", path)
    df = pd.read_csv(path)
    LOGGER.info("Data shape: %s", df.shape)
    return df


def make_X_y(df: pd.DataFrame, cfg: TrainConfig) -> tuple[pd.DataFrame, pd.Series]:
    if cfg.target_column not in df.columns:
        raise KeyError(f"Target column '{cfg.target_column}' not in data.")

    missing_feats = [c for c in cfg.feature_columns if c not in df.columns]
    if missing_feats:
        raise KeyError(f"Feature columns missing from data: {missing_feats}")

    X = df[list(cfg.feature_columns)].copy()
    y = df[cfg.target_column].copy()

    LOGGER.info("Using %d features: %s", X.shape[1], list(cfg.feature_columns))
    return X, y


def build_models(cfg: TrainConfig):
    """Return a dict of model_name -> estimator."""
    models = {}

    # Linear model with scaling
    ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=cfg.random_state)),
        ]
    )
    models["ridge"] = ridge

    # Tree-based model (no scaling needed)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=cfg.random_state,
    )
    models["random_forest"] = rf

    return models


def eval_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def save_scatter_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    lims = [
        min(np.min(y_true), np.min(y_pred)),
        max(np.max(y_true), np.max(y_pred)),
    ]
    plt.plot(lims, lims)  # y=x line
    plt.xlabel("True LN_IC50")
    plt.ylabel("Predicted LN_IC50")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    LOGGER.info("Saved scatter plot to %s", out_path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    args = parse_args()

    cfg = TrainConfig(
        data_path=args.data_path,
        out_dir=args.out_dir,
        target_column=args.target_column,
    )
    LOGGER.info("Training with config: %s", cfg)

    df = load_model_data(cfg.data_path)
    X, y = make_X_y(df, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )
    LOGGER.info("Train size: %d, Test size: %d", X_train.shape[0], X_test.shape[0])

    models = build_models(cfg)
    metrics_rows = []

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        LOGGER.info("Fitting model: %s", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = eval_model(y_test, y_pred)
        metrics["model"] = name
        metrics_rows.append(metrics)

        # Save predictions
        preds_df = pd.DataFrame(
            {"y_true": y_test.values, "y_pred": y_pred},
            index=y_test.index,
        )
        preds_path = cfg.out_dir / f"{name}_ytrue_ypred.csv"
        preds_df.to_csv(preds_path, index=False)
        LOGGER.info("Saved predictions to %s", preds_path)

        # Scatter plot
        plot_path = cfg.out_dir / f"{name}_scatter.png"
        save_scatter_plot(
            y_true=y_test.values,
            y_pred=y_pred,
            out_path=plot_path,
            title=f"{name} - True vs Predicted LN_IC50",
        )

    # Save metrics table
    metrics_df = pd.DataFrame(metrics_rows).set_index("model")
    metrics_path = cfg.out_dir / "baseline_metrics.csv"
    metrics_df.to_csv(metrics_path)
    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()

