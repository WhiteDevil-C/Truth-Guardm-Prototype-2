import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend" / "data"
OUT = ROOT / "backend" / "experiments"
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
    valid = texts.ne("") & labels.ne("")
    return texts[valid], labels[valid]

def main():
    print("Loading train and validation data...")
    X_train_text, y_train = load_split("train")
    X_valid_text, y_valid = load_split("valid")

    print(f"Training examples: {len(y_train)}")
    print(f"Validation examples: {len(y_valid)}")

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_valid = vectorizer.transform(X_valid_text)

    models = {
        "logistic_regression_C1": LogisticRegression(
            solver="lbfgs",
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        "logistic_regression_C5": LogisticRegression(
            solver="lbfgs",
            C=5.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        "linear_svc": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
        ),
        "multinomial_nb": MultinomialNB(alpha=0.1),
    }

    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)

        accuracy = accuracy_score(y_valid, predictions)
        macro_f1 = f1_score(y_valid, predictions, average="macro")
        weighted_f1 = f1_score(y_valid, predictions, average="weighted")

        result = {
            "model": name,
            "accuracy": round(float(accuracy), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
        }
        results.append(result)

        print(
            f"Accuracy={accuracy:.4f} | "
            f"Macro-F1={macro_f1:.4f} | "
            f"Weighted-F1={weighted_f1:.4f}"
        )
        print(classification_report(y_valid, predictions, zero_division=0))

    results.sort(key=lambda row: row["macro_f1"], reverse=True)
    results_path = OUT / "validation_comparison.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n===== VALIDATION LEADERBOARD =====")
    for row in results:
        print(
            f'{row["model"]:28} '
            f'accuracy={row["accuracy"]:.4f} '
            f'macro_f1={row["macro_f1"]:.4f} '
            f'weighted_f1={row["weighted_f1"]:.4f}'
        )

    print(f"\nResults saved to: {results_path}")
    print("No existing model artifacts were modified.")
    print("test.tsv was not used.")

if __name__ == "__main__":
    main()
