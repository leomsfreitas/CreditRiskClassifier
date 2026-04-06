from sklearn.ensemble import RandomForestClassifier

MODEL = RandomForestClassifier(
    class_weight="balanced",
    n_jobs=-1,
)

MODEL_NAME = "RandomForest"