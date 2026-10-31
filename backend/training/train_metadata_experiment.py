import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MaxAbsScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend" / "data"
OUT = ROOT / "backend" / "experiments" / "metadata_model"
OUT.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "statement_id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count", "context",
]

CAT_COLS = ["subject", "speaker", "job_title", "state_info", "party_affiliation"]
NUM_COLS = [
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count",
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
    df["label"] = df["label"].fillna("").astype(str).str.strip()

    for col in CAT_COLS:
        df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()
        df.loc[df[col].eq(""), col] = "unknown"

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df[df["statement"].ne("") & df["label"].ne("")].copy()

def metadata_dicts(df):
    records = []
    for row in df.to_dict(orient="records"):
        item = {f"{col}={row[col]}": 1.0 for col in CAT_COLS}
        item.update({col: float(row[col]) for col in NUM_COLS})
        records.append(item)
    return records

def metrics(y_true, predictions):
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "macro_f1": round(float(f1_score(y_true, predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, predictions, average="weighted")), 4),
    }

def main():
    print("Loading train and validation splits only...")
    train = load_split("train")
    valid = load_split("valid")

    word_vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_text = word_vectorizer.fit_transform(train["statement"])
    X_valid_text = word_vectorizer.transform(valid["statement"])

    meta_vectorizer = DictVectorizer(sparse=True)
    X_train_meta = meta_vectorizer.fit_transform(metadata_dicts(train))
    X_valid_meta = meta_vectorizer.transform(metadata_dicts(valid))

    # Scale numeric/categorical metadata features as a group so raw counts
    # do not dominate the text TF-IDF features.
    scaler = MaxAbsScaler()
    X_train_meta = scaler.fit_transform(X_train_meta)
    X_valid_meta = scaler.transform(X_valid_meta)

    results = []
    best_model = None
    best_accuracy = -1.0
    best_macro_f1 = -1.0

    for use_metadata in [False, True]:
        name = "text_plus_metadata" if use_metadata else "text_only"
        if use_metadata:
            X_train = hstack([X_train_text, X_train_meta], format="csr")
            X_valid = hstack([X_valid_text, X_valid_meta], format="csr")
        else:
            X_train, X_valid = X_train_text, X_valid_text

        for c_value in [1.0, 5.0]:
            print(f"\nTraining {name}, C={c_value}...")
            model = LogisticRegression(
                solver="lbfgs",
                C=c_value,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )
            model.fit(X_train, train["label"])
            predictions = model.predict(X_valid)
            score = metrics(valid["label"], predictions)

            result = {
                "model": name,
                "C": c_value,
                "validation_samples": len(valid),
                **score,
            }
            results.append(result)
            print(json.dumps(result, indent=2))

            if (score["accuracy"], score["macro_f1"]) > (
                best_accuracy, best_macro_f1
            ):
                best_accuracy = score["accuracy"]
                best_macro_f1 = score["macro_f1"]
                best_model = model
                best_name = name
                best_C = c_value

    # Save experiment metrics, winning classifier, and preprocessing.
    (OUT / "validation_metrics.json").write_text(
        json.dumps({
            "selection_rule": "highest validation accuracy, then macro-F1",
            "results": results,
            "best_model": best_name,
            "best_C": best_C,
            "test_split_used": False,
            "caution": (
                "Speaker and historical metadata may introduce bias or "
                "memorization; this experiment is not a verified fact-checker."
            ),
        }, indent=2),
        encoding="utf-8",
    )

    joblib.dump(best_model, OUT / "classifier.pkl")
    joblib.dump(word_vectorizer, OUT / "word_vectorizer.pkl")
    joblib.dump(meta_vectorizer, OUT / "metadata_vectorizer.pkl")
    joblib.dump(scaler, OUT / "metadata_scaler.pkl")

    config = {
        "model": best_name,
        "C": best_C,
        "labels": [str(label) for label in best_model.classes_],
        "text_features": {
            "max_features": 30000,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "min_df": 2
        },
        "metadata_features": {
            "categorical_columns": CAT_COLS,
            "numeric_columns": NUM_COLS,
            "enabled": best_name == "text_plus_metadata"
        },
        "validation_accuracy": best_accuracy,
        "validation_macro_f1": best_macro_f1,
        "test_split_used": False,
        "production_model_changed": False
    }
    (OUT / "model_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nWinning model:", best_name, "| C =", best_C)
    print("Validation accuracy:", best_accuracy)
    print("Saved experiment artifacts to:", OUT)
    print("Production artifacts untouched; test.tsv was not loaded.")

if __name__ == "__main__":
    main()
