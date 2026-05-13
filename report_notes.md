# Report Notes for NeurIPS-Style Final Paper

## Abstract draft

Student dropout is a major challenge for higher education institutions because delayed identification of at-risk students limits the time available for academic and financial support. This project studies whether machine learning models can predict dropout risk using student demographic, socioeconomic, enrollment, and first-year academic performance features. Using the UCI Predict Students' Dropout and Academic Success dataset, I build a reproducible sklearn pipeline with one-hot encoding, binary feature mapping, standardized numeric features, engineered academic progress variables, and strict train/test separation. I compare a majority-class baseline, logistic regression, random forest, and XGBoost on both binary dropout detection and three-class academic outcome prediction. Results show that tree-based ensemble models outperform the baseline and logistic regression, while feature-importance analysis identifies second-semester approvals, approval rates, tuition status, and academic progress measures as key predictors. These findings suggest that interpretable machine learning can support early-warning systems, although responsible use requires careful attention to fairness, calibration, and human oversight.

## Introduction points

- Real-world problem: universities need to identify students at risk of dropping out early enough to intervene.
- Prediction target: binary dropout risk and multiclass final academic outcome.
- Why data science is appropriate: tabular institutional data contains demographic, socioeconomic, and academic performance signals.
- Contribution: complete reproducible pipeline, engineered academic progress features, baseline comparison, ensemble model comparison, and interpretability analysis.

## Methodology points

- Train/test split: stratified 80/20.
- Preprocessing is fit only on the training set inside sklearn pipelines.
- One-hot encoded categorical integer-code fields.
- Passed through 0/1 binary fields after validation.
- Scaled continuous variables using StandardScaler.
- Engineered approval rates and grade-change features.
- Compared majority baseline, logistic regression, random forest, and XGBoost.

## Experiments section points

Include:

- Target distribution figure.
- Binary model metrics table.
- Multiclass model metrics table.
- Confusion matrix for best binary model.
- ROC curve for binary task.
- PCA projection.
- Feature importance or SHAP plot.

## Limitations

- Dataset comes from one institution/country, so generalization to other universities is uncertain.
- Some features, especially first- and second-semester performance, may only be available after students have already begun struggling.
- Predictive systems can create fairness risks if used punitively or without student support resources.
- The project does not perform subgroup fairness analysis; this should be future work.
