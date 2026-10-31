import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend" / "data"
OUTPUT = ROOT / "backend" / "experiments" / "calibration_comparison.json"

COLUMNS = [
    "statement_id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count", "context",
]

def load_split(name):
    df = pd.read_csv(
        DATA / f"{name}.tsv",
        sep="\t",
        header=None,
        names=COLUMNS,
        quoting=3,
        on_bad_lines="skip",
    )
    df["statement"] = (
        df["statement"].fillna("").astype(str).str.strip().str.lower()
    )
    df["label"] = df["label"].astype(str).str.strip()
    return df

train = load_split("train")
valid = load_split("valid")

# Recreate the saved model's training configuration.
base_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )),
    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
    )),
])

X_train, y_train = train["statement"], train["label"]
X_valid, y_valid = valid["statement"], valid["label"]

print("Training baseline on training split...")
base_model.fit(X_train, y_train)

models = {
    "uncalibrated": base_model,
}

for method in ("sigmoid", "isotonic"):
    print(f"Fitting {method} calibration...")
    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method=method,
        cv=3,
        ensemble=True,
    )
    calibrated.fit(X_train, y_train)
    models[method] = calibrated

results = []

for name, model in models.items():
    probabilities = model.predict_proba(X_valid)
    predictions = model.classes_[probabilities.argmax(axis=1)]

    result = {
        "model": name,
        "validation_samples": len(y_valid),
        "accuracy": round(float(accuracy_score(y_valid, predictions)), 4),
        "macro_f1": round(float(f1_score(
            y_valid, predictions, average="macro", zero_division=0
        )), 4),
        "weighted_f1": round(float(f1_score(
            y_valid, predictions, average="weighted", zero_division=0
        )), 4),
        "log_loss": round(float(log_loss(
            y_valid, probabilities, labels=model.classes_
        )), 4),
    }
    results.append(result)
    print(json.dumps(result, indent=2))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

print(f"\nSaved comparison to: {OUTPUT}")
print("Production artifacts and held-out test metrics were not changed.")
