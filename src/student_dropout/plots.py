from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .config import FIGURES_DIR, TARGET_COL
from .features import get_feature_names


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_target_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET_COL].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar")
    plt.title("Target class distribution")
    plt.xlabel("Student outcome")
    plt.ylabel("Number of students")
    _savefig(FIGURES_DIR / "target_distribution.png")


def plot_numeric_correlation(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include=[np.number])
    # Keep strongest correlations with the binary dropout indicator to make the plot legible.
    temp = numeric.copy()
    temp["Dropout indicator"] = (df[TARGET_COL] == "Dropout").astype(int)
    corr = temp.corr(numeric_only=True)
    top = corr["Dropout indicator"].abs().sort_values(ascending=False).head(18).index
    corr_top = corr.loc[top, top]

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_top, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(top)), top, rotation=90, fontsize=7)
    plt.yticks(range(len(top)), top, fontsize=7)
    plt.title("Correlation among strongest numeric dropout-related features")
    _savefig(FIGURES_DIR / "correlation_heatmap.png")


def plot_metrics_bar(metrics: pd.DataFrame, task: str) -> None:
    cols = ["accuracy", "f1_macro", "roc_auc"]
    available = [col for col in cols if col in metrics.columns]
    plot_df = metrics.set_index("model")[available]
    plot_df.plot(kind="bar", figsize=(9, 4))
    plt.ylim(0, 1)
    plt.title(f"{task.capitalize()} model comparison")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=25, ha="right")
    _savefig(FIGURES_DIR / f"{task}_model_metrics.png")


def plot_confusion_matrix(result: dict) -> None:
    task = result["task"]
    model = result["best_model"]
    X_test = result["X_test"]
    y_test = result["y_test"]
    y_pred = model.predict(X_test)

    plt.figure(figsize=(5, 5))
    display_labels = None
    if task == "binary":
        display_labels = ["Not dropout", "Dropout"]
    elif result.get("label_encoder") is not None:
        display_labels = result["label_encoder"].classes_
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=display_labels, xticks_rotation=35)
    plt.title(f"Best {task} model confusion matrix: {result['best_name']}")
    _savefig(FIGURES_DIR / f"{task}_confusion_matrix.png")


def plot_binary_roc_curves(result: dict) -> None:
    if result["task"] != "binary":
        return
    X_test = result["X_test"]
    y_test = result["y_test"]
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    for name, model in result["models"].items():
        if hasattr(model, "predict_proba"):
            RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    plt.title("Binary dropout ROC curves")
    _savefig(FIGURES_DIR / "binary_roc_curves.png")


def plot_pca_projection(result: dict) -> None:
    task = result["task"]
    model = result["best_model"]
    preprocessor = model.named_steps["preprocess"]
    X_test = result["X_test"]
    y_test = result["y_test_raw"] if task == "multiclass" else result["y_test"]
    X_processed = preprocessor.transform(X_test)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    coords = PCA(n_components=2, random_state=42).fit_transform(X_processed)
    plt.figure(figsize=(7, 5))
    labels = pd.Series(y_test).astype(str)
    for label in sorted(labels.unique()):
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.7, label=label)
    plt.title(f"PCA projection of preprocessed {task} test data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Label", fontsize=8)
    _savefig(FIGURES_DIR / f"{task}_pca_projection.png")


def plot_random_forest_feature_importance(result: dict) -> None:
    if "random_forest" not in result["models"]:
        return
    task = result["task"]
    model = result["models"]["random_forest"]
    rf = model.named_steps["model"]
    preprocessor = model.named_steps["preprocess"]
    names = get_feature_names(preprocessor)
    importances = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False).head(20)

    plt.figure(figsize=(9, 6))
    importances.sort_values().plot(kind="barh")
    plt.title(f"Top random forest feature importances ({task})")
    plt.xlabel("Mean decrease in impurity")
    _savefig(FIGURES_DIR / f"{task}_random_forest_feature_importance.png")


def plot_shap_importance(result: dict) -> None:
    """Create SHAP bar plot for XGBoost if possible, otherwise silently skip.

    SHAP is not essential for the pipeline to run. The project still produces feature
    importances and permutation-ready fitted models if SHAP is unavailable.
    """
    if "xgboost" not in result["models"]:
        return

    try:
        import shap
    except Exception:
        return

    task = result["task"]
    model = result["models"]["xgboost"]
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    X_sample = result["X_test"].sample(min(350, len(result["X_test"])), random_state=42)
    X_processed = preprocessor.transform(X_sample)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    names = get_feature_names(preprocessor)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_processed)

    plt.figure(figsize=(9, 6))
    if isinstance(shap_values, list):
        values = np.mean([np.abs(v) for v in shap_values], axis=0)
    else:
        values = np.abs(shap_values)
        if values.ndim == 3:  # multiclass: samples x features x classes
            values = values.mean(axis=2)
    mean_abs = pd.Series(values.mean(axis=0), index=names).sort_values(ascending=False).head(20)
    mean_abs.sort_values().plot(kind="barh")
    plt.title(f"Mean absolute SHAP importance for XGBoost ({task})")
    plt.xlabel("Mean |SHAP value|")
    _savefig(FIGURES_DIR / f"{task}_xgboost_shap_importance.png")


def generate_all_plots(df: pd.DataFrame, binary_result: dict, multiclass_result: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_target_distribution(df)
    plot_numeric_correlation(df)
    for result in [binary_result, multiclass_result]:
        plot_metrics_bar(result["metrics"], result["task"])
        plot_confusion_matrix(result)
        plot_pca_projection(result)
        plot_random_forest_feature_importance(result)
        plot_shap_importance(result)
    plot_binary_roc_curves(binary_result)
