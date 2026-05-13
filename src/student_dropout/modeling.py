from __future__ import annotations

from typing import Literal

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .features import build_preprocessor


def build_models(task: Literal["binary", "multiclass"] = "binary") -> dict[str, Pipeline]:
    """Create baseline and main model pipelines.

    All non-dummy models use the same preprocessing pipeline, which prevents leakage
    because the transformers are fit only inside the training split.
    """
    preprocessor = build_preprocessor

    models: dict[str, Pipeline] = {
        "majority_baseline": Pipeline(
            steps=[("preprocess", preprocessor()), ("model", DummyClassifier(strategy="most_frequent"))]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if task == "binary":
        models["xgboost"] = Pipeline(
            steps=[
                ("preprocess", preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=120,
                        max_depth=3,
                        learning_rate=0.07,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    elif task == "multiclass":
        models["xgboost"] = Pipeline(
            steps=[
                ("preprocess", preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        n_estimators=120,
                        max_depth=3,
                        learning_rate=0.07,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    else:
        raise ValueError("task must be either 'binary' or 'multiclass'")

    return models
