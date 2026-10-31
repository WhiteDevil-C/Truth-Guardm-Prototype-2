import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend" / "data"
OUT = ROOT / "backend" / "experiments" / "tuned_model"
OUT.mkdir(parents=True, exist_ok=True)

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
    texts = df["statement"].fillna("").astype(str).str.strip().str.lower()
    labels = df["label"].fillna("").astype(str).str.strip()
    mask = texts.ne("") & labels.ne("")
    return texts[mask], labels[mask]

def main():
    train_text, y_train = load_split("train")
    valid_text, y_valid = load_split("valid")

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )

    X_train = vectorizer.fit_transform(train_text)
    X_valid = vectorizer.transform(valid_text)

    classifier = LogisticRegression(
        solver="lbfgs",
        C=5.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_valid)
    metrics = {
        "validation_samples": len(y_valid),
        "accuracy": round(float(accuracy_score(y_valid, predictions)), 4),
        "macro_f1": round(float(f1_score(
            y_valid, predictions, average="macro"
        )), 4),
        "weighted_f1": round(float(f1_score(
            y_valid, predictions, average="weighted"
        )), 4),
        "configuration": {
            "max_features": 30000,
            "ngram_range": [1, 2],
            "C": 5.0,
            "class_weight": "balanced",
        },
    }

    joblib.dump(vectorizer, OUT / "vectorizer.pkl")
    joblib.dump(classifier, OUT / "classifier.pkl")

    (OUT / "labels.json").write_text(
        json.dumps(list(classifier.classes_), indent=2),
        encoding="utf-8",
    )
    (OUT / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("Tuned model trained and saved separately.")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts: {OUT}")
    print("Original backend/model files were not modified.")
    print("test.tsv was not loaded.")

if __name__ == "__main__":
    main()
