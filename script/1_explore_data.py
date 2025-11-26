#!/usr/bin/env python3
"""
01_explore_data.py

Exploratory Data Analysis (EDA) for the GDSC dataset.

This script is written to be part of a research-grade ML project:
- Uses a config dataclass
- Uses pathlib for robust paths
- Uses logging instead of print
- Can be called from the command line or imported as a module
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------------------------

LOGGER = logging.getLogger("cdr_ml.eda")

# This file lives in: project_root / script / 01_explore_data.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

@dataclass
class EDAConfig:
    """Configuration for EDA run."""

    data_path: Path = PROJECT_ROOT / "data" / "GDSC_DATASET.csv"
    out_dir: Path = PROJECT_ROOT / "results" / "eda"


# -----------------------------------------------------------------------------
# CLI ARGUMENTS & LOGGING
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run basic EDA on the GDSC drug-response dataset."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=EDAConfig().data_path,
        help="Path to the GDSC dataset CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EDAConfig().out_dir,
        help="Directory to write EDA outputs.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Configure logging for script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


# -----------------------------------------------------------------------------
# CORE LOGIC
# -----------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    """Load the GDSC dataset from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    LOGGER.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    LOGGER.info("Dataset loaded with shape %s", df.shape)
    return df


def build_summary(df: pd.DataFrame) -> str:
    """Assemble a multi-section text summary."""
    lines: list[str] = []

    lines.append("=== DATASET OVERVIEW ===")
    lines.append(f"Shape (rows, columns): {df.shape[0]} rows, {df.shape[1]} columns\n")

    lines.append("=== COLUMNS ===")
    lines.append(", ".join(df.columns.tolist()))
    lines.append("")

    lines.append("=== DATA TYPES (first 40 columns) ===")
    lines.append(df.dtypes.astype(str).head(40).to_string())
    lines.append("")

    lines.append("=== MISSING VALUES (top 30 columns by NA count) ===")
    na_counts = df.isna().sum().sort_values(ascending=False).head(30)
    lines.append(na_counts.to_string())
    lines.append("")

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        lines.append("=== NUMERIC SUMMARY (first 20 features) ===")
        desc = numeric.describe().T.head(20)
        lines.append(desc.to_string())
        lines.append("")
    else:
        lines.append("No numeric columns detected.\n")

    lines.append("=== FIRST 5 ROWS ===")
    lines.append(df.head().to_string())
    lines.append("")

    return "\n".join(lines)


def run_eda(config: EDAConfig) -> Path:
    """
    Run EDA using the given config.

    Returns
    -------
    Path
        Path to the written summary file.
    """
    config.out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output directory: %s", config.out_dir)

    df = load_dataset(config.data_path)
    summary_text = build_summary(df)

    out_file = config.out_dir / "eda_summary.txt"
    out_file.write_text(summary_text)

    LOGGER.info("EDA summary written to %s", out_file)
    return out_file


# -----------------------------------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    args = parse_args()

    config = EDAConfig(
        data_path=args.data_path,
        out_dir=args.out_dir,
    )

    LOGGER.info("Starting EDA with config: %s", config)
    run_eda(config)


if __name__ == "__main__":
    main()

