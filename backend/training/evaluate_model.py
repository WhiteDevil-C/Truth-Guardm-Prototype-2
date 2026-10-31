"""
TruthGuard — Held-out Test Set Evaluation
==========================================
Evaluates the CURRENT saved model artifacts (backend/model/*.pkl) against
backend/data/test.tsv, which train_model.py never touches during training.

This script does not retrain or tune anything. It only measures the already
-saved model so the numbers in the README reflect the actual shipped
artifacts, not a cherry-picked run.

Run from the project root:
    python backend/training/evaluate_model.py
"""

import os
import sys
import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "backend", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "model")
TEST_PATH = os.path.join(DATA_DIR, "test.tsv")

COLUMN_NAMES = [
    "statement_id", "label", "statement", "subject", "speaker", "job_title",
    "state_info", "party_affiliation", "barely_true_count", "false_count",
    "half_true_count", "mostly_true_count", "pants_fire_count", "context",
]


def main():
    if not os.path.exists(TEST_PATH):
        print(f"ERROR: Cannot find test.tsv at:\n  {TEST_PATH}")
        print("Place the LIAR dataset at backend/data/ (see README) and retry.")
        sys.exit(1)

    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    classifier_path = os.path.join(MODEL_DIR, "classifier.pkl")
    if not os.path.exists(vectorizer_path) or not os.path.exists(classifier_path):
        print("ERROR: Trained model artifacts not found in backend/model/.")
        print("Run backend/training/train_model.py first.")
        sys.exit(1)

    test_df = pd.read_csv(
        TEST_PATH, sep="\t", header=None, names=COLUMN_NAMES,
        quoting=3, on_bad_lines="skip",
    )
    X_test = test_df["statement"].fillna("").astype(str).str.strip().str.lower()
    y_test = test_df["label"].str.strip()

    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)

    matrix = vectorizer.transform(X_test)
    predictions = classifier.predict(matrix)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")
    report = classification_report(y_test, predictions, digits=4)

    print("=" * 65)
    print(f"  TruthGuard — Held-out TEST set evaluation ({len(test_df)} samples)")
    print("=" * 65)
    print(f"\nAccuracy:     {accuracy:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}\n")
    print(report)

    print(
        "NOTE: This is a 6-class classification task over short political "
        "claims (the LIAR dataset). Random guessing would score ~16.7%. "
        "These numbers should be read as 'meaningfully better than chance, "
        "far from reliable fact-checking' — not as production-grade accuracy."
    )

    results_path = os.path.join(MODEL_DIR, "test_metrics.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "test_samples": len(test_df),
                "accuracy": round(float(accuracy), 4),
                "macro_f1": round(float(macro_f1), 4),
                "weighted_f1": round(float(weighted_f1), 4),
            },
            fh,
            indent=2,
        )
    print(f"\nSaved summary metrics to {results_path}")


if __name__ == "__main__":
    main()
