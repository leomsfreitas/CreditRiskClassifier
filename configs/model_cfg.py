from xgboost import XGBClassifier

MODEL = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

MODEL_NAME = "xgboost"