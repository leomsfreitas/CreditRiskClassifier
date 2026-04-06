import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

from preprocessing import DataPreprocessor
from configs.model_cfg import MODEL, MODEL_NAME
import configs.data_cfg as data_cfg

MODEL_PATH   = Path("models/model.joblib")
METRICS_PATH = Path("reports/metrics.json")
FULL_REPORT_PATH = Path("reports/full_report.json")

def train():
    df = DataPreprocessor.clean(
        pd.read_csv("data/raw/amex_data.csv", index_col=data_cfg.INDEX_COL),
        drop_cols=data_cfg.DROP_COLS,
        filter_values=data_cfg.FILTER_VALUES
    )
    
    X = df.drop(columns=[data_cfg.TARGET_COL])
    y = df[data_cfg.TARGET_COL]
        
    X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            stratify=y,
            test_size=0.2,
            random_state=2026
        )

    ordinal_transformer = FunctionTransformer(
        DataPreprocessor.apply_ordinal,
        kw_args={"categories": data_cfg.ORDINAL_ORDER},
        validate=False,
        feature_names_out="one-to-one"
    )

    map_transformer = FunctionTransformer(
        DataPreprocessor.apply_map,
        kw_args={"map_dict": data_cfg.MAP_DICT},
        validate=False,
        feature_names_out="one-to-one"
    )

    ohe_transformer = FunctionTransformer(
        DataPreprocessor.apply_ohe,
        kw_args={"dtype": "int8"},
        validate=False
    )

    cast_transformer = FunctionTransformer(
        DataPreprocessor.cast_numeric,
        kw_args={"numeric_type": "float32"},
        validate=False,
        feature_names_out="one-to-one"
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
        remainder="passthrough"
    )
    
    train_pipe = Pipeline(
        steps=[
            (MODEL_NAME, MODEL)
        ]
    )
    
    complete_pipe = Pipeline(
        steps=[
            ("preprocess", prep_pipe),
            ("train", train_pipe),
        ]
    )
    
    model = MODEL
    complete_pipe.fit(X_train, y_train)
    
    y_pred = complete_pipe.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        "model": MODEL_NAME,
        "accuracy": round(report["accuracy"], 4),
        "f1": round(report["macro avg"]["f1-score"], 4),
        "precision": round(report["macro avg"]["precision"], 4),
        "recall": round(report["macro avg"]["recall"], 4),
    }
    
    full_report = {
        "model": MODEL_NAME,
        **report,
    }
    
    print(metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FULL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(complete_pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    FULL_REPORT_PATH.write_text(json.dumps(full_report, indent=2))

if __name__ == "__main__":
    train()