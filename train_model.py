from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "behavior_data.csv"
MODEL_PATH = ROOT / "bot_detector.pkl"
FEATURES = ["Distance", "Time", "Jitter"]
TARGET = "Is_Human"


def main():
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = data[required].dropna()
    if set(data[TARGET].unique()) != {0, 1}:
        raise ValueError("Is_Human must contain both binary labels 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET].astype(int),
        test_size=0.2,
        random_state=42,
        stratify=data[TARGET],
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Held-out classification report:")
    print(classification_report(y_test, predictions, target_names=["Bot (0)", "Human (1)"]))
    print("Confusion matrix [rows=true, columns=predicted]:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
