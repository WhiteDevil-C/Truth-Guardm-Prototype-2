import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend" / "data"
OUT = ROOT / "backend" / "experiments" / "metadata_model"

COLUMNS = [
    "statement_id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count", "context",
]
CAT_COLS = [
    "subject", "speaker", "job_title", "state_info", "party_affiliation"
]
NUM_COLS = [
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count",
]

def metadata_dicts(df):
    records = []
    for row in df.to_dict(orient="records"):
        item = {
            f"{col}={row[col]}": 1.0
            for col in CAT_COLS
        }
        item.update({col: float(row[col]) for col in NUM_COLS})
        records.append(item)
    return records

def score(y_true, predictions):
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "macro_f1": round(float(f1_score(
            y_true, predictions, average="macro", zero_division=0
        )), 4),
        "weighted_f1": round(float(f1_score(
            y_true, predictions, average="weighted", zero_division=0
        )), 4),
    }

print("Loading validation split only...")
df = pd.read_csv(
    DATA / "valid.tsv",
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

df = df[df["statement"].ne("") & df["label"].ne("")].copy()

model = joblib.load(OUT / "classifier.pkl")
word_vectorizer = joblib.load(OUT / "word_vectorizer.pkl")
meta_vectorizer = joblib.load(OUT / "metadata_vectorizer.pkl")
scaler = joblib.load(OUT / "metadata_scaler.pkl")

X_text = word_vectorizer.transform(df["statement"])

# Scenario A: metadata available, as in the original validation experiment.
X_meta_real = scaler.transform(
    meta_vectorizer.transform(metadata_dicts(df))
)
X_real = hstack([X_text, X_meta_real], format="csr")
pred_real = model.predict(X_real)

# Scenario B: extension-like input with unknown speaker/context metadata.
unknown_df = df.copy()
for col in CAT_COLS:
    unknown_df[col] = "unknown"
for col in NUM_COLS:
    unknown_df[col] = 0

X_meta_unknown = scaler.transform(
    meta_vectorizer.transform(metadata_dicts(unknown_df))
)
X_unknown = hstack([X_text, X_meta_unknown], format="csr")
pred_unknown = model.predict(X_unknown)

results = {
    "validation_samples": len(df),
    "with_real_dataset_metadata": score(df["label"], pred_real),
    "with_unknown_metadata": score(df["label"], pred_unknown),
    "accuracy_difference_percentage_points": round(
        (accuracy_score(df["label"], pred_unknown)
         - accuracy_score(df["label"], pred_real)) * 100, 2
    ),
    "macro_f1_difference": round(
        f1_score(df["label"], pred_unknown, average="macro", zero_division=0)
        - f1_score(df["label"], pred_real, average="macro", zero_division=0),
        4
    ),
    "test_split_used": False,
    "production_model_changed": False,
}

print(json.dumps(results, indent=2))
(OUT / "metadata_availability_evaluation.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
print("\nSaved report to:", OUT / "metadata_availability_evaluation.json")
print("Production artifacts untouched; test.tsv was not loaded.")
