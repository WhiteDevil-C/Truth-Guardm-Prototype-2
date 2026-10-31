import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend" / "data"
OUT = ROOT / "backend" / "experiments" / "word_char_model"
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
    print("Loading train and validation splits only...")
    train_text, y_train = load_split("train")
    valid_text, y_valid = load_split("valid")

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        max_features=30000,
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=2,
        dtype="float32",
    )

    print("Fitting word features...")
    X_train_word = word_vectorizer.fit_transform(train_text)
    X_valid_word = word_vectorizer.transform(valid_text)

    print("Fitting character features...")
    X_train_char = char_vectorizer.fit_transform(train_text)
    X_valid_char = char_vectorizer.transform(valid_text)

    print("Combining sparse feature matrices...")
    X_train = hstack(
        [X_train_word, X_train_char], format="csr", dtype="float32"
    )
    X_valid = hstack(
        [X_valid_word, X_valid_char], format="csr", dtype="float32"
    )

    results = []
    best_model = None
    best_accuracy = -1.0
    best_macro_f1 = -1.0

    for c_value in [1.0, 5.0]:
        print(f"\nTraining Logistic Regression, C={c_value}...")
        model = LogisticRegression(
            solver="lbfgs",
            C=c_value,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)

        accuracy = float(accuracy_score(y_valid, predictions))
        macro_f1 = float(f1_score(y_valid, predictions, average="macro"))
        weighted_f1 = float(
            f1_score(y_valid, predictions, average="weighted")
        )

        result = {
            "model": "word_char_tfidf_logistic_regression",
            "C": c_value,
            "validation_samples": len(y_valid),
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "word_features": X_train_word.shape[1],
            "character_features": X_train_char.shape[1],
        }
        results.append(result)

        print(json.dumps(result, indent=2))

        # Primary objective: accuracy; macro-F1 breaks ties.
        if (accuracy, macro_f1) > (best_accuracy, best_macro_f1):
            best_accuracy = accuracy
            best_macro_f1 = macro_f1
            best_model = model

    joblib.dump(word_vectorizer, OUT / "word_vectorizer.pkl")
    joblib.dump(char_vectorizer, OUT / "char_vectorizer.pkl")
    joblib.dump(best_model, OUT / "classifier.pkl")

    (OUT / "labels.json").write_text(
        json.dumps(list(best_model.classes_), indent=2),
        encoding="utf-8",
    )
    (OUT / "validation_metrics.json").write_text(
        json.dumps({
            "selection_rule": "highest validation accuracy, then macro-F1",
            "results": results,
            "best_C": best_model.C,
            "test_split_used": False,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nSaved separate word + character experiment to:", OUT)
    print("Production model untouched.")
    print("test.tsv was not loaded.")

if __name__ == "__main__":
    main()
