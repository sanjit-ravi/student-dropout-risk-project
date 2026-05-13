from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from student_dropout.config import DATA_PATH, OUTPUT_DIR, TABLES_DIR
from student_dropout.data import make_dataset
from student_dropout.evaluate import train_and_evaluate
from student_dropout.plots import generate_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the student dropout risk prediction project.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH, help="Path to semicolon-delimited UCI data.csv")
    parser.add_argument("--skip-plots", action="store_true", help="Train/evaluate without creating figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    X_binary, y_binary, clean_df = make_dataset(args.data_path, task="binary")
    X_multi, y_multi, _ = make_dataset(args.data_path, task="multiclass")

    print("Dataset loaded")
    print(f"Rows: {len(clean_df):,}")
    print(f"Features after engineering: {X_binary.shape[1]}")
    print("Target distribution:")
    print(clean_df["Target"].value_counts().to_string())

    print("\nTraining binary dropout-risk models...")
    binary_result = train_and_evaluate(X_binary, y_binary, task="binary")
    print(binary_result["metrics"].round(4).to_string(index=False))
    print(f"Best binary model: {binary_result['best_name']}")

    print("\nTraining multiclass student-outcome models...")
    multiclass_result = train_and_evaluate(X_multi, y_multi, task="multiclass")
    print(multiclass_result["metrics"].round(4).to_string(index=False))
    print(f"Best multiclass model: {multiclass_result['best_name']}")

    if not args.skip_plots:
        print("\nCreating figures...")
        generate_all_plots(clean_df, binary_result, multiclass_result)

    summary = {
        "rows": int(len(clean_df)),
        "raw_features": 36,
        "features_after_engineering": int(X_binary.shape[1]),
        "binary_best_model": binary_result["best_name"],
        "binary_best_metrics": binary_result["metrics"].iloc[0].to_dict(),
        "multiclass_best_model": multiclass_result["best_name"],
        "multiclass_best_metrics": multiclass_result["metrics"].iloc[0].to_dict(),
    }
    with open(TABLES_DIR / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
