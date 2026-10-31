import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend" / "data"
EXP = ROOT / "backend" / "experiments"
META = EXP / "metadata_model"
TUNED = EXP / "tuned_model"
OUT = EXP / "text_vs_metadata_comparison.json"

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
        item = {f"{c}={row[c]}": 1.0 for c in CAT_COLS}
        item.update({c: float(row[c]) for c in NUM_COLS})
        records.append(item)
    return records

def evaluate(y_true, predictions, labels):
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "macro_f1": round(float(f1_score(
            y_true, predictions, labels=labels,
            average="macro", zero_division=0
        )), 4),
        "weighted_f1": round(float(f1_score(
            y_true, predictions, labels=labels,
            average="weighted", zero_division=0
        )), 4),
        "per_class": classification_report(
            y_true, predictions, labels=labels,
            output_dict=True, zero_division=0
        ),
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

for c in CAT_COLS:
    df[c] = df[c].fillna("unknown").astype(str).str.strip().str.lower()
    df.loc[df[c].eq(""), c] = "unknown"
for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

df = df[df["statement"].ne("") & df["label"].ne("")].copy()
y = df["label"]
labels = sorted(y.unique())

# Candidate 1: tuned text-only model.
text_model = joblib.load(TUNED / "classifier.pkl")
text_vectorizer = joblib.load(TUNED / "vectorizer.pkl")
X_text = text_vectorizer.transform(df["statement"])
pred_text = text_model.predict(X_text)

# Candidate 2: metadata model with metadata deliberately unknown.
meta_model = joblib.load(META / "classifier.pkl")
meta_text_vectorizer = joblib.load(META / "word_vectorizer.pkl")
meta_vectorizer = joblib.load(META / "metadata_vectorizer.pkl")
scaler = joblib.load(META / "metadata_scaler.pkl")

X_meta_text = meta_text_vectorizer.transform(df["statement"])
unknown = df.copy()
for c in CAT_COLS:
    unknown[c] = "unknown"
for c in NUM_COLS:
    unknown[c] = 0

X_unknown_meta = scaler.transform(
    meta_vectorizer.transform(metadata_dicts(unknown))
)
X_meta = hstack([X_meta_text, X_unknown_meta], format="csr")
pred_meta = meta_model.predict(X_meta)

results = {
    "validation_samples": len(df),
    "labels": labels,
    "text_only_tuned": evaluate(y, pred_text, labels),
    "metadata_model_unknown_metadata": evaluate(y, pred_meta, labels),
    "test_split_used": False,
    "production_model_changed": False,
    "note": (
        "Validation comparison only. Neither model verifies factual truth. "
        "Metadata model uses unknown metadata for the second scenario."
    ),
}

OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

for name in ("text_only_tuned", "metadata_model_unknown_metadata"):
    r = results[name]
    print(f"\n{name}")
    print("Accuracy:", r["accuracy"])
    print("Macro F1:", r["macro_f1"])
    print("Weighted F1:", r["weighted_f1"])
    print("\nPer-class report:")
    for label in labels:
        row = r["per_class"][label]
        print(
            f"  {label:12s} precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} f1={row['f1-score']:.3f}"
        )

print("\nSaved report:", OUT)
print("Production artifacts untouched; test.tsv was not loaded.")
