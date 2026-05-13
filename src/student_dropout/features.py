from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    ALL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_BINARY_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    ORDINAL_NUMERIC_FEATURES,
    CONTINUOUS_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing pipeline required by the rubric.

    - Nominal integer codes are one-hot encoded.
    - Binary variables are passed through after explicit 0/1 validation in data.py.
    - Continuous/ordinal variables and engineered numeric variables are standardized.
    """
    numeric_features = ORDINAL_NUMERIC_FEATURES + CONTINUOUS_FEATURES + ENGINEERED_NUMERIC_FEATURES
    binary_features = BINARY_FEATURES + ENGINEERED_BINARY_FEATURES

    return ColumnTransformer(
        transformers=[
            ("categorical_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL_FEATURES),
            ("numeric_scaled", StandardScaler(), numeric_features),
            ("binary_passthrough", "passthrough", binary_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return readable feature names after fitting the ColumnTransformer."""
    names = preprocessor.get_feature_names_out().tolist()
    cleaned = []
    for name in names:
        for prefix in ["categorical_ohe__", "numeric_scaled__", "binary_passthrough__"]:
            name = name.replace(prefix, "")
        cleaned.append(name)
    return cleaned


def validate_feature_columns(columns: list[str] | np.ndarray) -> None:
    """Ensure all planned features exist before training."""
    missing = sorted(set(ALL_FEATURES) - set(columns))
    if missing:
        raise ValueError(f"Missing expected features: {missing}")
