"""
data_loader.py
--------------
Step 1 of the preprocessing pipeline.
Loads the HP WIP-A DCS Excel export, validates structure,
maps sensor tags, and returns a clean base DataFrame.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from config import (
    RAW_DATA_FILE, DATA_SHEET, TAGS_SHEET, DATE_COLUMN,
    ALL_SENSORS, SENSOR_LABELS, SAMPLING_INTERVAL_MIN
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: Path = RAW_DATA_FILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw Excel historian export.

    Returns
    -------
    df   : Main data DataFrame with Date index and 35 sensor columns.
    tags : Tags reference DataFrame (Tag Name → Tag Description).
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Place the Excel file in the data/ folder and update RAW_DATA_FILE in config.py"
        )

    logger.info(f"Loading raw data from: {filepath.name}")

    # Load both sheets
    df   = pd.read_excel(filepath, sheet_name=DATA_SHEET, header=0)
    tags = pd.read_excel(filepath, sheet_name=TAGS_SHEET)

    # Drop unnamed index column that Excel sometimes exports
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # Parse dates
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # Force all sensor columns to numeric — coerce any residual strings
    for col in df.columns:
        if col != DATE_COLUMN:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    logger.info(f"Date range: {df[DATE_COLUMN].min()} → {df[DATE_COLUMN].max()}")

    return df, tags


def validate_structure(df: pd.DataFrame) -> None:
    """
    Verify the DataFrame has the expected sensor columns
    and timestamps are consistent.
    """
    logger.info("Validating data structure...")

    # Check all expected sensors are present
    missing = [s for s in ALL_SENSORS if s not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")

    # Check timestamp regularity
    diffs = df[DATE_COLUMN].diff().dropna().dt.total_seconds() / 60
    expected_interval = SAMPLING_INTERVAL_MIN
    irregular = (diffs != expected_interval).sum()

    if irregular > 0:
        logger.warning(f"{irregular} irregular timestamp intervals detected (expected {expected_interval} min)")
    else:
        logger.info(f"All timestamps validated at {expected_interval}-minute intervals")

    # Missing value report
    total_cells = len(df) * len(ALL_SENSORS)
    missing_vals = df[ALL_SENSORS].isnull().sum()
    total_missing = missing_vals.sum()
    missing_pct = (total_missing / total_cells) * 100

    logger.info(f"Missing values: {total_missing:,} / {total_cells:,} ({missing_pct:.3f}%)")

    if missing_pct > 5.0:
        logger.warning(f"Missing value rate ({missing_pct:.1f}%) exceeds 5% — review data quality")

    sensors_with_missing = missing_vals[missing_vals > 0]
    if len(sensors_with_missing) > 0:
        logger.info(f"Sensors with missing values:\n{sensors_with_missing.to_string()}")


def summarise_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary statistics DataFrame for all sensor channels.
    Useful for EDA and initial inspection.
    """
    from config import FAULT_ONSET_DATE, VIBRATION_SENSORS, TEMPERATURE_SENSORS

    normal  = df[df[DATE_COLUMN] < FAULT_ONSET_DATE]
    fault   = df[df[DATE_COLUMN] >= FAULT_ONSET_DATE]

    rows = []
    for col in ALL_SENSORS:
        sensor_type = "Vibration" if col in VIBRATION_SENSORS else "Temperature"
        rows.append({
            "Tag":          col,
            "Description":  SENSOR_LABELS.get(col, col),
            "Type":         sensor_type,
            "Normal Mean":  round(normal[col].mean(), 2),
            "Normal Std":   round(normal[col].std(), 2),
            "Fault Mean":   round(fault[col].mean(), 2),
            "Fault Std":    round(fault[col].std(), 2),
            "Overall Min":  round(df[col].min(), 2),
            "Overall Max":  round(df[col].max(), 2),
            "Missing":      df[col].isnull().sum(),
        })

    summary = pd.DataFrame(rows)

    # Calculate % change between normal and fault mean
    summary["Mean Change %"] = (
        (summary["Fault Mean"] - summary["Normal Mean"]) / summary["Normal Mean"] * 100
    ).round(1)

    return summary


if __name__ == "__main__":
    df, tags = load_raw_data()
    validate_structure(df)
    summary = summarise_data(df)
    print("\n=== DATA SUMMARY ===")
    print(summary[["Description", "Type", "Normal Mean", "Fault Mean", "Mean Change %", "Missing"]].to_string())
