#!/usr/bin/env python3
"""
02_prepare_dataset.py

Prepare a clean modelling dataset from the GDSC summary table.

- Target: LN_IC50 (drug sensitivity).
- Features: simple numeric features (COSMIC_ID, DRUG_ID, AUC, Z_SCORE).
- Meta: identifiers we keep for analysis/plots (DRUG_NAME, CELL_LINE_NAME, etc.).

Output:
- results/processed/model_data.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("cdr_ml.prepare")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------

@dataclass
class PrepareConfig:
    data_path: Path = PROJECT_ROOT / "data" / "GDSC_DATASET.csv"
    out_dir: Path = PROJECT_ROOT / "results" / "processed"

    # Target we want to predict (from your EDA)
    target_column: str = "LN_IC50"

    # Simple numeric features
    numeric_features: tuple[str, ...] = ("COSMIC_ID", "DRUG_ID", "AUC", "Z_SCORE")

    # Meta columns to keep for later inspection/plots (not used as X now)
    meta_columns: tuple[str, ...] = (
        "CELL_LINE_NAME",
        "DRUG_NAME",
        "TCGA_DESC",
        "GDSC Tissue descriptor 1",
        "GDSC Tissue descriptor 2",
        "Cancer Type (matching TCGA label)",
        "Screen Medium",
        "Growth Properties",
        "TARGET",
        "TARGET_PATHWAY",
    )


# -------------------------------------------------------------------------
# ARGPARSE & LOGGING
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare modelling-ready dataset from GDSC summary table."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PrepareConfig().data_path,
        help="Path to GDSC_DATASET.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PrepareConfig().out_dir,
        help="Directory to write processed dataset.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=PrepareConfig().target_column,
        help="Target column name (default: LN_IC50).",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


# -------------------------------------------------------------------------
# CORE LOGIC
# -------------------------------------------------------------------------

def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    LOGGER.info("Loading raw table from %s", path)
    df = pd.read_csv(path)
    LOGGER.info("Raw shape: %s", df.shape)
    return df


def prepare_dataset(cfg: PrepareConfig, df: pd.DataFrame) -> pd.DataFrame:
    # Check target
    if cfg.target_column not in df.columns:
        raise KeyError(f"Target column '{cfg.target_column}' not found in table.")

    # Keep only columns that exist
    numeric_feats = [c for c in cfg.numeric_features if c in df.columns]
    meta_cols = [c for c in cfg.meta_columns if c in df.columns]

    cols_to_keep = [cfg.target_column] + numeric_feats + meta_cols
    df_sub = df[cols_to_keep].copy()
    LOGGER.info("Subset columns: %d (target + %d numeric + %d meta)",
                1, len(numeric_feats), len(meta_cols))

    # Drop rows with missing target or numeric features
    before = len(df_sub)
    df_sub = df_sub.dropna(subset=[cfg.target_column] + numeric_feats)
    after = len(df_sub)
    LOGGER.info("Dropped %d rows with NA in target/numeric features (kept %d).",
                before - after, after)

    return df_sub


def save_dataset(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_data.csv"
    df.to_csv(out_path, index=False)
    LOGGER.info("Saved processed dataset to %s (shape=%s)", out_path, df.shape)
    return out_path


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    args = parse_args()

    cfg = PrepareConfig(
        data_path=args.data_path,
        out_dir=args.out_dir,
        target_column=args.target_column,
    )
    LOGGER.info("Running dataset preparation with config: %s", cfg)

    df_raw = load_table(cfg.data_path)
    df_model = prepare_dataset(cfg, df_raw)
    save_dataset(df_model, cfg.out_dir)


if __name__ == "__main__":
    main()

