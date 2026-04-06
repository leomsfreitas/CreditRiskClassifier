import sys
from pathlib import Path
import json
import joblib

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

sys.path.append(str(Path(__file__).parent.parent))

from preprocessing import DataPreprocessor
from src.train import train
from configs.model_cfg import MODEL, MODEL_NAME
import configs.data_cfg as data_cfg


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Gina", "Hank"],
        "age": [25.0, 35.0, 45.0, 55.0, 65.0, 29.0, 41.0, 52.0],
        "owns_car": ["Y", "N", "Y", "N", "Y", "N", "Y", "N"],
        "owns_house": ["N", "Y", "Y", "N", "Y", "N", "N", "Y"],
        "no_of_children": [0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 2.0, 0.0],
        "net_yearly_income": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0, 52000.0, 68000.0, 79000.0],
        "no_of_days_employed": [100.0, 200.0, 300.0, 400.0, 500.0, 120.0, 260.0, 380.0],
        "occupation_type": [
            "Managers",
            "Laborers",
            "Drivers",
            "Accountants",
            "IT staff",
            "Sales staff",
            "Core staff",
            "Managers",
        ],
        "total_family_members": [1.0, 2.0, 3.0, 1.0, 4.0, 2.0, 3.0, 1.0],
        "migrant_worker": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "gender": ["F", "M", "F", "M", "F", "M", "F", "M"],
        "credit_card_default": [0, 1, 0, 1, 0, 1, 0, 1],
    })


def build_complete_pipeline(X: pd.DataFrame) -> Pipeline:
    ordinal_transformer = FunctionTransformer(
        DataPreprocessor.apply_ordinal,
        kw_args={"categories": data_cfg.ORDINAL_ORDER},
        validate=False,
        feature_names_out="one-to-one",
    )

    map_transformer = FunctionTransformer(
        DataPreprocessor.apply_map,
        kw_args={"map_dict": data_cfg.MAP_DICT},
        validate=False,
        feature_names_out="one-to-one",
    )

    ohe_transformer = FunctionTransformer(
        DataPreprocessor.apply_ohe,
        validate=False,
    )

    cast_transformer = FunctionTransformer(
        DataPreprocessor.cast_numeric,
        kw_args={"numeric_type": "float32"},
        validate=False,
        feature_names_out="one-to-one",
    )

    prep_pipe = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                Pipeline([
                    ("ordinal", ordinal_transformer),
                    ("cast", cast_transformer),
                ]),
                data_cfg.ORDINAL_COLS,
            ),
            (
                "mapped",
                Pipeline([
                    ("map", map_transformer),
                    ("cast", cast_transformer),
                ]),
                data_cfg.MAP_COLS,
            ),
            (
                "ohe",
                ohe_transformer,
                data_cfg.NOMINAL_COLS,
            ),
            (
                "numeric",
                cast_transformer,
                X.columns,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", prep_pipe),
            (MODEL_NAME, MODEL),
        ]
    )


def test_pipeline_fit_predict(sample_df):
    df = DataPreprocessor.clean(
        sample_df,
        drop_cols=data_cfg.DROP_COLS,
        filter_values=data_cfg.FILTER_VALUES,
    )

    X = df.drop(columns=[data_cfg.TARGET_COL])
    y = df[data_cfg.TARGET_COL]

    pipe = build_complete_pipeline(X)
    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_train_generates_artifacts(monkeypatch, sample_df, tmp_path):
    model_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    import src.train as train_module

    monkeypatch.setattr(train_module.pd, "read_csv", lambda *_args, **_kwargs: sample_df.copy())
    monkeypatch.setattr(train_module, "MODEL_PATH", model_dir / "model.joblib")
    monkeypatch.setattr(train_module, "METRICS_PATH", reports_dir / "metrics.json")
    monkeypatch.setattr(train_module, "FULL_REPORT_PATH", reports_dir / "full_report.json")

    train()

    assert (model_dir / "model.joblib").exists()
    assert (reports_dir / "metrics.json").exists()
    assert (reports_dir / "full_report.json").exists()

    metrics = json.loads((reports_dir / "metrics.json").read_text())
    assert "model" in metrics
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics


def test_saved_artifact_can_be_loaded_and_predict(monkeypatch, sample_df, tmp_path):
    model_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    import src.train as train_module

    monkeypatch.setattr(train_module.pd, "read_csv", lambda *_args, **_kwargs: sample_df.copy())
    monkeypatch.setattr(train_module, "MODEL_PATH", model_dir / "model.joblib")
    monkeypatch.setattr(train_module, "METRICS_PATH", reports_dir / "metrics.json")
    monkeypatch.setattr(train_module, "FULL_REPORT_PATH", reports_dir / "full_report.json")

    train()

    loaded_model = joblib.load(model_dir / "model.joblib")

    df = DataPreprocessor.clean(
        sample_df,
        drop_cols=data_cfg.DROP_COLS,
        filter_values=data_cfg.FILTER_VALUES,
    )
    X = df.drop(columns=[data_cfg.TARGET_COL])

    preds = loaded_model.predict(X)

    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})