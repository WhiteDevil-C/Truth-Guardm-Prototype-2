import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend" / "data"
OUT = ROOT / "backend" / "experiments" / "tuned_model" / "sigmoid_candidate"
OUT.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "statement_id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count", "context",
]

def main():
    print("Loading training split...")
    df = pd.read_csv(
        DATA / "train.tsv",
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

    print(f"Training samples: {len(labels)}")

    print("Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train = vectorizer.fit_transform(texts)

    base_model = LogisticRegression(
        solver="lbfgs",
        C=5.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    print("Training sigmoid calibration with 3-fold CV...")
    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=3,
        ensemble=False,
    )
    calibrated.fit(X_train, labels)

    # Preserve the fitted linear estimator for the existing XAI logic.
    xai_model = calibrated.calibrated_classifiers_[0].estimator

    assert list(calibrated.classes_) == list(xai_model.classes_)
    assert hasattr(xai_model, "coef_")

    print("Saving candidate files...")
    joblib.dump(vectorizer, OUT / "vectorizer.pkl")
    joblib.dump(calibrated, OUT / "calibrated_classifier.pkl")
    joblib.dump(xai_model, OUT / "xai_classifier.pkl")

    metadata = {
        "model": "tuned_word_tfidf_logistic_regression_sigmoid",
        "calibration": "sigmoid",
        "calibration_cv": 3,
        "ensemble": False,
        "C": 5.0,
        "class_weight": "balanced",
        "training_samples": len(labels),
        "test_split_used": False,
        "production_model_modified": False,
    }

    (OUT / "candidate_info.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    sample = X_train[:1]
    probabilities = calibrated.predict_proba(sample)[0]

    assert abs(float(probabilities.sum()) - 1.0) < 1e-6

    print("\n===== CANDIDATE CHECK =====")
    print("Prediction:", calibrated.predict(sample)[0])
    print("Classes:", list(calibrated.classes_))
    print("Probability sum:", round(float(probabilities.sum()), 6))
    print("XAI coefficient shape:", xai_model.coef_.shape)
    print("Saved to:", OUT)
    print("Production model untouched.")
    print("Validation and test data not loaded.")

if __name__ == "__main__":
    main()
