import pandas as pd
from sklearn.model_selection import train_test_split

from student_dropout.config import ALL_FEATURES, DATA_PATH
from student_dropout.data import make_dataset
from student_dropout.features import build_preprocessor
from student_dropout.modeling import build_models


def test_dataset_loads_and_engineers_features():
    X, y, df = make_dataset(DATA_PATH, task="binary")
    assert len(df) == 4424
    assert X.shape[0] == y.shape[0]
    assert set(y.unique()) == {0, 1}
    assert set(ALL_FEATURES).issubset(set(X.columns))
    assert int(df.isna().sum().sum()) == 0


def test_preprocessor_fits_without_leakage():
    X, y, _ = make_dataset(DATA_PATH, task="binary")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = build_preprocessor()
    Xt = preprocessor.fit_transform(X_train)
    assert Xt.shape[0] == X_train.shape[0]
    assert preprocessor.transform(X_test).shape[0] == X_test.shape[0]


def test_logistic_pipeline_smoke_fit():
    X, y, _ = make_dataset(DATA_PATH, task="binary")
    sample = X.sample(500, random_state=42)
    y_sample = y.loc[sample.index]
    model = build_models("binary")["logistic_regression"]
    model.fit(sample, y_sample)
    preds = model.predict(sample.head(25))
    assert len(preds) == 25
