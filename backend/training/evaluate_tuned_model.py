import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend" / "data"
MODEL = ROOT / "backend" / "experiments" / "tuned_model"
TEST = DATA / "test.tsv"

COLUMNS = [
    "statement_id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count", "context",
]

def main():
    if not TEST.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST}")

    df = pd.read_csv(
        TEST,
        sep="\t",
        header=None,
        names=COLUMNS,
        quoting=3,
        on_bad_lines="skip",
    )

    texts = df["statement"].fillna("").astype(str).str.strip().str.lower()
    labels = df["label"].fillna("").astype(str).str.strip()
    mask = texts.ne("") & labels.ne("")
    texts, labels = texts[mask], labels[mask]

    vectorizer = joblib.load(MODEL / "vectorizer.pkl")
    classifier = joblib.load(MODEL / "classifier.pkl")

    predictions = classifier.predict(vectorizer.transform(texts))

    metrics = {
        "test_samples": len(labels),
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(
            labels, predictions, average="macro"
        )), 4),
        "weighted_f1": round(float(f1_score(
            labels, predictions, average="weighted"
        )), 4),
    }

    print("HELD-OUT TEST RESULTS")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(classification_report(labels, predictions, zero_division=0))

    output = MODEL / "test_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Results saved to: {output}")
    print("Do not use this test result to tune further configurations.")

if __name__ == "__main__":
    main()
