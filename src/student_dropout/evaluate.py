from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import MODELS_DIR, RANDOM_STATE, TABLES_DIR, TEST_SIZE
from .features import validate_feature_columns
from .modeling import build_models


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 train/test split."""
    validate_feature_columns(X.columns)
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def _roc_auc(task: str, y_true, proba, classes=None) -> float:
    if proba is None:
        return float("nan")
    try:
        if task == "binary":
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted"))
    except Exception:
        return float("nan")


def evaluate_model(name: str, model, X_test, y_test, task: str) -> dict:
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    average = "binary" if task == "binary" else "macro"
    pos_label = 1 if task == "binary" else None

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": _roc_auc(task, y_test, proba),
    }

    if task == "binary":
        metrics.update(
            {
                "dropout_precision": precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
                "dropout_recall": recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
                "dropout_f1": f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
            }
        )

    return metrics


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, task: Literal["binary", "multiclass"]):
    """Train all models and return fitted models plus metrics."""
    X_train, X_test, y_train_raw, y_test_raw = split_data(X, y)

    label_encoder = None
    y_train = y_train_raw
    y_test = y_test_raw
    if task == "multiclass":
        # XGBoost requires integer class labels; keep encoder for interpretation.
        label_encoder = LabelEncoder()
        y_train = pd.Series(label_encoder.fit_transform(y_train_raw), index=y_train_raw.index, name=y_train_raw.name)
        y_test = pd.Series(label_encoder.transform(y_test_raw), index=y_test_raw.index, name=y_test_raw.name)

    models = build_models(task)
    fitted_models = {}
    metrics_rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        metrics_rows.append(evaluate_model(name, model, X_test, y_test, task))

    metrics = pd.DataFrame(metrics_rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted_models[best_name]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(TABLES_DIR / f"{task}_metrics.csv", index=False)

    y_pred = best_model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    with open(TABLES_DIR / f"{task}_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    with open(TABLES_DIR / f"{task}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm).to_csv(TABLES_DIR / f"{task}_confusion_matrix.csv", index=False)

    joblib.dump(best_model, MODELS_DIR / f"best_{task}_model.joblib")
    if label_encoder is not None:
        joblib.dump(label_encoder, MODELS_DIR / f"{task}_label_encoder.joblib")

    return {
        "task": task,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
        "models": fitted_models,
        "metrics": metrics,
        "best_name": best_name,
        "best_model": best_model,
        "label_encoder": label_encoder,
    }
