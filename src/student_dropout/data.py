from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .config import (
    BINARY_FEATURES,
    DATA_PATH,
    DROPOUT_POSITIVE_LABEL,
    TARGET_COL,
)


def load_raw_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw UCI student dropout dataset.

    The original CSV uses semicolon delimiters. Column names are cleaned only for
    accidental tabs/trailing spaces so the feature names remain recognizable in the report.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the UCI data.csv file there or pass --data-path."
        )
    df = pd.read_csv(path, sep=";")
    df.columns = [col.replace("\t", "").strip() for col in df.columns]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative data cleaning and validation.

    The UCI page reports no missing values, but this function still removes duplicate rows,
    validates the target, and casts binary variables to integers for reproducibility.
    """
    df = df.copy()
    df.columns = [col.replace("\t", "").strip() for col in df.columns]

    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}', found: {df.columns.tolist()}")

    # Drop exact duplicates only. No imputation is required for the released dataset.
    df = df.drop_duplicates().reset_index(drop=True)

    missing_total = int(df.isna().sum().sum())
    if missing_total > 0:
        raise ValueError(
            f"Unexpected missing values found ({missing_total}). Add imputation before training."
        )

    expected_targets = {"Dropout", "Enrolled", "Graduate"}
    actual_targets = set(df[TARGET_COL].unique())
    if not actual_targets.issubset(expected_targets):
        raise ValueError(f"Unexpected target labels: {sorted(actual_targets)}")

    for col in BINARY_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing expected binary feature: {col}")
        values = set(df[col].dropna().unique())
        if not values.issubset({0, 1}):
            raise ValueError(f"Binary feature {col!r} contains values other than 0/1: {values}")
        df[col] = df[col].astype(int)

    return df


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two Series and replace divide-by-zero cases with zero."""
    result = numerator.astype(float) / denominator.replace(0, np.nan).astype(float)
    return result.fillna(0.0)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create academically meaningful features from first/second semester performance."""
    df = df.copy()

    first_enrolled = df["Curricular units 1st sem (enrolled)"]
    second_enrolled = df["Curricular units 2nd sem (enrolled)"]
    first_approved = df["Curricular units 1st sem (approved)"]
    second_approved = df["Curricular units 2nd sem (approved)"]

    df["Total curricular units enrolled"] = first_enrolled + second_enrolled
    df["Total curricular units approved"] = first_approved + second_approved
    df["Overall approval rate"] = safe_divide(
        df["Total curricular units approved"], df["Total curricular units enrolled"]
    )
    df["1st sem approval rate"] = safe_divide(first_approved, first_enrolled)
    df["2nd sem approval rate"] = safe_divide(second_approved, second_enrolled)
    df["Grade change 2nd minus 1st sem"] = (
        df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
    )
    df["Admission minus previous qualification grade"] = (
        df["Admission grade"] - df["Previous qualification (grade)"]
    )
    df["No 1st sem approvals"] = (first_approved == 0).astype(int)
    df["No 2nd sem approvals"] = (second_approved == 0).astype(int)
    return df


def make_dataset(
    path: str | Path = DATA_PATH,
    task: Literal["binary", "multiclass"] = "binary",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return features X, target y, and the cleaned full dataframe.

    Binary task maps Dropout to 1 and Enrolled/Graduate to 0.
    Multiclass task leaves labels as Dropout, Enrolled, Graduate.
    """
    df = add_engineered_features(clean_data(load_raw_data(path)))
    X = df.drop(columns=[TARGET_COL])

    if task == "binary":
        y = (df[TARGET_COL] == DROPOUT_POSITIVE_LABEL).astype(int)
        y.name = "Dropout risk"
    elif task == "multiclass":
        y = df[TARGET_COL].copy()
    else:
        raise ValueError("task must be either 'binary' or 'multiclass'")

    return X, y, df
