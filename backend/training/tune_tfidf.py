import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

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
    text = df["statement"].fillna("").astype(str).str.strip().str.lower()
    labels = df["label"].fillna("").astype(str).str.strip()
    mask = text.ne("") & labels.ne("")
    return text[mask], labels[mask]

def main():
    print("Loading training and validation datasets...")
    train_text, y_train = load_split("train")
    valid_text, y_valid = load_split("valid")

    results = []

    for features in [20000, 30000, 50000]:
        for ngrams in [(1, 2), (1, 3)]:
            print(f"\nPreparing TF-IDF: features={features}, ngrams={ngrams}")

            vectorizer = TfidfVectorizer(
                max_features=features,
                ngram_range=ngrams,
                sublinear_tf=True,
                min_df=2,
            )

            X_train = vectorizer.fit_transform(train_text)
            X_valid = vectorizer.transform(valid_text)

            for c_value in [5.0, 10.0]:
                print(f"Training Logistic Regression C={c_value}...")

                model = LogisticRegression(
                    solver="lbfgs",
                    C=c_value,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                predictions = model.predict(X_valid)

                result = {
                    "max_features": features,
                    "ngram_range": list(ngrams),
                    "C": c_value,
                    "accuracy": round(
                        float(accuracy_score(y_valid, predictions)), 4
                    ),
                    "macro_f1": round(
                        float(f1_score(y_valid, predictions, average="macro")), 4
                    ),
                    "weighted_f1": round(
                        float(f1_score(y_valid, predictions, average="weighted")), 4
                    ),
                    "vocabulary_size": len(vectorizer.vocabulary_),
                }
                results.append(result)

                print(
                    f"Accuracy={result['accuracy']:.4f} | "
                    f"Macro-F1={result['macro_f1']:.4f} | "
                    f"Weighted-F1={result['weighted_f1']:.4f}"
                )

    results.sort(key=lambda row: row["macro_f1"], reverse=True)
    output = OUT / "tfidf_tuning_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n===== TOP 10 VALIDATION CONFIGURATIONS =====")
    for rank, row in enumerate(results[:10], start=1):
        print(
            f"{rank}. features={row['max_features']}, "
            f"ngrams={row['ngram_range']}, C={row['C']} | "
            f"accuracy={row['accuracy']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}, "
            f"weighted_f1={row['weighted_f1']:.4f}"
        )

    print(f"\nFull results saved to: {output}")
    print("Existing model artifacts were not modified.")
    print("test.tsv was not loaded or used.")

if __name__ == "__main__":
    main()
