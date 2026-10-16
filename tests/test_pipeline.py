"""Smoke tests for data generation, training and inference."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn import predict as predict_module
from churn.config import FEATURES, TARGET
from churn.data import generate
from churn.pipeline import build_pipeline


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return generate(n=1000, seed=0)


@pytest.fixture(scope="module")
def model_path(data, tmp_path_factory):
    import joblib

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(data[FEATURES], data[TARGET])
    path = tmp_path_factory.mktemp("models") / "model.joblib"
    joblib.dump({"name": "test", "pipeline": pipeline, "features": FEATURES}, path)
    return path


def test_generate_has_expected_schema(data):
    assert len(data) == 1000
    assert set(FEATURES + [TARGET]) <= set(data.columns)
    assert data[TARGET].isin([0, 1]).all()
    assert 0.05 < data[TARGET].mean() < 0.5


def test_generate_is_reproducible():
    pd.testing.assert_frame_equal(generate(n=50, seed=1), generate(n=50, seed=1))


def test_predict_adds_scores(data, model_path, monkeypatch):
    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    out = predict_module.predict(data.head(20))
    assert out["churn_probability"].between(0, 1).all()
    assert out["churn_prediction"].isin([0, 1]).all()


def test_predict_is_independent_of_batch_composition(data, model_path, monkeypatch):
    """A row must get the same score whether scored alone or with other rows."""
    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    batch = predict_module.predict(data.head(50))
    single = predict_module.predict(data.iloc[[7]])
    assert single["churn_probability"].iloc[0] == batch["churn_probability"].iloc[7]


def test_predict_rejects_missing_columns(data, model_path, monkeypatch):
    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    with pytest.raises(ValueError, match="missing required columns"):
        predict_module.predict(data.drop(columns=["tenure_months"]).head(5))
