import json
from pathlib import Path

import joblib
from scipy.sparse import hstack

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "backend" / "experiments" / "metadata_model"

config = json.loads((OUT / "model_config.json").read_text(encoding="utf-8"))

model = joblib.load(OUT / "classifier.pkl")
word_vectorizer = joblib.load(OUT / "word_vectorizer.pkl")
metadata_vectorizer = joblib.load(OUT / "metadata_vectorizer.pkl")
scaler = joblib.load(OUT / "metadata_scaler.pkl")

claim = "The government announced a new economic policy today."

metadata = {
    "subject": "unknown",
    "speaker": "unknown",
    "job_title": "unknown",
    "state_info": "unknown",
    "party_affiliation": "unknown",
    "barely_true_count": 0,
    "false_count": 0,
    "half_true_count": 0,
    "mostly_true_count": 0,
    "pants_fire_count": 0,
}

# Match the training script's normalization and feature construction.
claim = claim.strip().lower()
cat_cols = config["metadata_features"]["categorical_columns"]
num_cols = config["metadata_features"]["numeric_columns"]

metadata_record = {
    f"{col}={str(metadata[col]).strip().lower()}": 1.0
    for col in cat_cols
}
metadata_record.update({
    col: float(metadata[col]) for col in num_cols
})

X_text = word_vectorizer.transform([claim])

if config["metadata_features"]["enabled"]:
    X_meta = metadata_vectorizer.transform([metadata_record])
    X_meta = scaler.transform(X_meta)
    X = hstack([X_text, X_meta], format="csr")
else:
    X = X_text

prediction = str(model.predict(X)[0])
probabilities = model.predict_proba(X)[0]

print("Experimental model inference smoke test")
print("Model:", config["model"])
print("Prediction:", prediction)
print("Highest class score:", round(float(probabilities.max()), 4))
print("Probability scores:", {
    str(label): round(float(score), 4)
    for label, score in zip(model.classes_, probabilities)
})

assert prediction in config["labels"], "Unexpected prediction label"
assert X.shape[1] == model.n_features_in_, (
    f"Feature mismatch: got {X.shape[1]}, expected {model.n_features_in_}"
)

print("PASS: saved artifacts loaded and prediction completed.")
print("Note: this is a pipeline check, not verification of the claim's truth.")
