import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline

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
    text = df["statement"].fillna("").astype(str).str.strip().str.lower()
    labels = df["label"].fillna("").astype(str).str.strip()
    mask = text.ne("") & labels.ne("")
    return text[mask], labels[mask]

def expected_calibration_error(y_true, probabilities, classes, bins=10):
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = np.array(classes)[predictions] == np.asarray(y_true)

    ece = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        mask = (confidence >= lower) & (
            confidence < upper if upper < 1 else confidence <= upper
        )
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)

def evaluate(name, model, X_valid, y_valid, classes):
    probabilities = model.predict_proba(X_valid)
    predictions = model.predict(X_valid)

    return {
        "model": name,
        "validation_samples": len(y_valid),
        "accuracy": round(float(accuracy_score(y_valid, predictions)), 4),
        "macro_f1": round(float(f1_score(y_valid, predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_valid, predictions, average="weighted")), 4),
        "log_loss": round(float(log_loss(y_valid, probabilities, labels=classes)), 4),
        "multiclass_brier_score": round(
            float(np.mean(np.sum(
                (probabilities - pd.get_dummies(y_valid).reindex(
                    columns=classes, fill_value=0
                ).to_numpy()) ** 2,
                axis=1
            ))), 4
        ),
        "top_label_ece": round(
            expected_calibration_error(y_valid, probabilities, classes), 4
        ),
    }

def main():
    print("Loading training and validation data...")
    X_train, y_train = load_split("train")
    X_valid, y_valid = load_split("valid")

    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_valid)}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("classifier", LogisticRegression(
            solver="lbfgs",
            C=5.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    results = []

    print("\nTraining uncalibrated tuned model...")
    pipeline.fit(X_train, y_train)
    classes = pipeline.classes_
    results.append(evaluate(
        "uncalibrated", pipeline, X_valid, y_valid, classes
    ))

    for method in ["sigmoid", "isotonic"]:
        print(f"\nTraining {method} calibration (3-fold CV)...")
        calibrated = CalibratedClassifierCV(
            estimator=pipeline,
            method=method,
            cv=3,
            ensemble=False,
        )
        calibrated.fit(X_train, y_train)

        results.append(evaluate(
            method, calibrated, X_valid, y_valid, calibrated.classes_
        ))

    output = {
        "experiment": "tuned_word_only_tfidf_calibration",
        "configuration": {
            "max_features": 30000,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "min_df": 2,
            "logistic_regression_C": 5.0,
            "class_weight": "balanced",
            "calibration_cv": 3,
            "calibration_ensemble": False,
        },
        "training_samples": len(y_train),
        "validation_samples": len(y_valid),
        "test_split_used": False,
        "results": results,
    }

    path = OUT / "calibration_comparison.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n===== CALIBRATION COMPARISON =====")
    print(json.dumps(output, indent=2))
    print(f"\nResults saved to: {path}")
    print("Production model untouched. test.tsv was not loaded.")

if __name__ == "__main__":
    main()
