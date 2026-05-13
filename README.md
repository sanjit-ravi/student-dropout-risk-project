# Predicting Student Dropout Risk with Interpretable Machine Learning

This project implements a complete data science pipeline for predicting student dropout risk from the UCI **Predict Students' Dropout and Academic Success** dataset.

## Research question

Can machine learning models identify students at risk of dropping out using enrollment, demographic, socioeconomic, and first-year academic performance features?

## Dataset

- Source: UCI Machine Learning Repository, dataset ID 697.
- Records: 4,424 students.
- Raw features: 36.
- Target labels: `Dropout`, `Enrolled`, `Graduate`.
- License listed by UCI: Creative Commons Attribution 4.0 International.

The data file used by the code is located at:

```text
data/raw/data.csv
```

## Methods

The implementation includes:

1. Cleaning and validation
   - Removes exact duplicates.
   - Confirms no missing values.
   - Validates 0/1 binary features.
2. Feature engineering
   - Total enrolled units.
   - Total approved units.
   - First-semester, second-semester, and overall approval rates.
   - Grade change from first to second semester.
   - No-approval flags.
3. Preprocessing
   - One-hot encoding for integer-coded nominal features.
   - Standard scaling for continuous and ordinal numeric features.
   - 0/1 passthrough for binary variables.
   - All transformations occur inside sklearn pipelines after train/test split to prevent leakage.
4. Models
   - Majority-class baseline.
   - Logistic regression.
   - Random forest.
   - XGBoost.
5. Evaluation
   - Binary task: Dropout vs. Not Dropout.
   - Multiclass task: Dropout vs. Enrolled vs. Graduate.
   - Accuracy, macro precision, macro recall, macro F1, ROC-AUC.
   - Binary dropout precision, recall, and F1.
6. Visualizations
   - Target distribution.
   - Correlation heatmap.
   - Model comparison plots.
   - Confusion matrices.
   - ROC curves.
   - PCA projections.
   - Random forest feature importance.
   - XGBoost SHAP importance.

## How to run

From the project root:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python run_project.py
```

To run tests:

```bash
PYTHONPATH=src pytest -q
```

## Main outputs

After running, outputs are saved in:

```text
outputs/
  figures/
  models/
  tables/
```

Important files:

- `outputs/tables/binary_metrics.csv`
- `outputs/tables/multiclass_metrics.csv`
- `outputs/tables/results_summary.json`
- `outputs/models/best_binary_model.joblib`
- `outputs/models/best_multiclass_model.joblib`

