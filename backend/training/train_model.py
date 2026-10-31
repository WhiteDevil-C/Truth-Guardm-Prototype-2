"""
TruthGuard — Baseline Training Script
======================================
Model   : TF-IDF + Logistic Regression (6-class)
Dataset : LIAR (Politifact)  — Wang 2017
Split   : train.tsv → fit model
          valid.tsv → evaluate during development   ← used here
          test.tsv  → held out; NOT touched in this script

Run from the project root:
    python backend/training/train_model.py

Outputs (saved to backend/model/):
    vectorizer.pkl   — fitted TfidfVectorizer
    classifier.pkl   — fitted LogisticRegression
    labels.json      — ordered list of class names
"""

import os
import json
import sys

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve paths relative to this file so the script works regardless of
# which directory it is invoked from.
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_DIR  = os.path.join(PROJECT_ROOT, "backend", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "model")

TRAIN_PATH = os.path.join(DATA_DIR, "train.tsv")
VALID_PATH = os.path.join(DATA_DIR, "valid.tsv")
# test.tsv is listed here only to document its location; it is never loaded.
TEST_PATH  = os.path.join(DATA_DIR, "test.tsv")   # NOT used in this script

VECTORIZER_PATH  = os.path.join(MODEL_DIR, "vectorizer.pkl")
CLASSIFIER_PATH  = os.path.join(MODEL_DIR, "classifier.pkl")
LABELS_PATH      = os.path.join(MODEL_DIR, "labels.json")

# ---------------------------------------------------------------------------
# LIAR dataset column names
# The TSV files have no header row.
# Full schema (14 columns):
#   0  statement_id
#   1  label          ← target
#   2  statement      ← input text (this baseline uses only this column)
#   3  subject
#   4  speaker
#   5  job_title
#   6  state_info
#   7  party_affiliation
#   8  barely_true_count
#   9  false_count
#  10  half_true_count
#  11  mostly_true_count
#  12  pants_fire_count
#  13  context
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    "statement_id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party_affiliation",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
    "context",
]

# The six original LIAR labels, ordered from most true to most false.
# This ordering is used when printing the confusion matrix so rows/columns
# have a meaningful sequence rather than alphabetical order.
LABEL_ORDER = [
    "true",
    "mostly-true",
    "half-true",
    "barely-true",
    "false",
    "pants-fire",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_tsv(path: str) -> pd.DataFrame:
    """Load a LIAR TSV file and return a DataFrame with named columns."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,              # no header row in LIAR files
        names=COLUMN_NAMES,
        quoting=3,                # QUOTE_NONE — avoids issues with quotes in claims
        on_bad_lines="skip",      # skip malformed rows rather than crashing
    )
    return df


def preprocess_text(series: pd.Series) -> pd.Series:
    """
    Minimal, reproducible text preprocessing.

    Choices made (and why):
      - Lowercase:        Normalises surface variation ("False" vs "false",
                          "SAYS" vs "says") without losing information.
      - Strip whitespace: Removes leading/trailing whitespace from raw TSV values.
      - No stemming:      Stemming can conflate semantically distinct words
                          (e.g. "bank" from both "banking" and "bank").
                          TF-IDF handles morphological variation adequately
                          for a first baseline.
      - No stopword removal: Stopwords sometimes carry signal in political
                          claims ("did not", "never"). Removing them at this
                          stage could hurt recall. We let TF-IDF's IDF weighting
                          down-weight high-frequency words naturally.
      - No punctuation stripping: Punctuation can mark quotations and emphasis
                          in political claims. Keeping it preserves that signal
                          for the bigram features.

    If a value is NaN (missing statement), replace with empty string so the
    vectorizer does not crash.
    """
    return series.fillna("").astype(str).str.strip().str.lower()


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  TruthGuard — Baseline Training  (TF-IDF + Logistic Regression)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading dataset …")

    for path, label in [(TRAIN_PATH, "train.tsv"), (VALID_PATH, "valid.tsv")]:
        if not os.path.exists(path):
            print(f"\nERROR: Cannot find {label} at:\n  {path}")
            print("Make sure the LIAR dataset is placed at backend/data/")
            sys.exit(1)

    train_df = load_tsv(TRAIN_PATH)
    valid_df = load_tsv(VALID_PATH)

    print(f"  Training samples : {len(train_df):,}")
    print(f"  Validation samples: {len(valid_df):,}")

    # ------------------------------------------------------------------
    # 2. Extract features and labels
    # ------------------------------------------------------------------
    print("\n[2/6] Extracting text and labels …")

    X_train_raw = preprocess_text(train_df["statement"])
    y_train     = train_df["label"].str.strip()

    X_valid_raw = preprocess_text(valid_df["statement"])
    y_valid     = valid_df["label"].str.strip()

    # Warn if any unexpected labels appear in the training set
    unexpected = set(y_train.unique()) - set(LABEL_ORDER)
    if unexpected:
        print(f"  WARNING: Unexpected labels found in train.tsv: {unexpected}")
        print("  These will be present in the model but are not in LABEL_ORDER.")

    # Print label distribution in training data so class imbalance is visible
    print("\n  Label distribution in train.tsv:")
    counts = y_train.value_counts()
    for lbl in LABEL_ORDER:
        n = counts.get(lbl, 0)
        print(f"    {lbl:<15} {n:>5} samples")

    # ------------------------------------------------------------------
    # 3. TF-IDF vectorization
    # ------------------------------------------------------------------
    print("\n[3/6] Fitting TF-IDF vectorizer on training data …")

    # Parameter choices:
    #   max_features=20000  — vocabulary cap; keeps memory reasonable while
    #                         covering the most informative terms for ~10k
    #                         training samples. Increase to 50000 if accuracy
    #                         is limited by vocabulary size.
    #   ngram_range=(1, 2)  — unigrams + bigrams. Bigrams capture short phrases
    #                         that matter in political claims
    #                         ("did not", "no evidence", "mostly correct").
    #   sublinear_tf=True   — applies log(1 + tf) instead of raw tf, which
    #                         reduces the dominance of very frequent terms.
    #                         Standard best practice for short-text classification.
    #   min_df=2            — ignore terms that appear in only 1 document;
    #                         these are likely typos or unique names that won't
    #                         generalise to validation/test data.
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )

    X_train = vectorizer.fit_transform(X_train_raw)  # fit + transform on training
    X_valid = vectorizer.transform(X_valid_raw)       # transform only on validation

    print(f"  Vocabulary size : {len(vectorizer.vocabulary_):,} terms")
    print(f"  Training matrix : {X_train.shape}")
    print(f"  Validation matrix: {X_valid.shape}")

    # ------------------------------------------------------------------
    # 4. Train Logistic Regression
    # ------------------------------------------------------------------
    print("\n[4/6] Training Logistic Regression classifier …")

    # Parameter choices:
    #   multi_class='multinomial' — proper 6-class softmax loss, more principled
    #                               than one-vs-rest for overlapping classes.
    #   solver='lbfgs'            — efficient for dense multinomial problems;
    #                               well-suited to TF-IDF sparse matrices of
    #                               this size.
    #   C=1.0                     — default inverse regularisation strength.
    #                               If the model underfits, try C=5 or C=10.
    #                               If it overfits, try C=0.1.
    #                               We keep C=1.0 for the first baseline to
    #                               establish an untuned reference point.
    #   class_weight='balanced'   — adjusts weights inversely proportional to
    #                               class frequencies. The LIAR dataset is
    #                               moderately imbalanced; balanced weighting
    #                               prevents the model from ignoring minority
    #                               classes (pants-fire, true) in favour of the
    #                               majority (half-true, false).
    #   max_iter=1000             — enough iterations for convergence on this
    #                               vocabulary size; lbfgs sometimes needs more
    #                               than the default 100.
    #   random_state=42           — reproducibility.
    classifier = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    classifier.fit(X_train, y_train)
    print("  Training complete.")

    # ------------------------------------------------------------------
    # 5. Evaluate on validation set
    # ------------------------------------------------------------------
    print("\n[5/6] Evaluating on valid.tsv …")
    print("  (test.tsv is NOT used — held out for final evaluation)")

    y_pred = classifier.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    # classification_report prints per-class + macro + weighted averages.
    # - Macro average: unweighted mean across classes — treats each class
    #   equally regardless of size. Important for catching poor minority-class
    #   performance.
    # - Weighted average: weights by support — reflects overall performance
    #   on the actual distribution. Use this to compare against prior work
    #   on LIAR which typically reports weighted or macro F1.
    report = classification_report(
        y_valid,
        y_pred,
        labels=LABEL_ORDER,
        digits=4,
        zero_division=0,
    )

    # Confusion matrix with labels in LABEL_ORDER for readability
    cm = confusion_matrix(y_valid, y_pred, labels=LABEL_ORDER)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS — Validation Set")
    print("=" * 65)

    print(f"\n  Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

    print("\n  Per-class metrics + Macro/Weighted averages:")
    print("  (macro avg = unweighted mean across 6 classes;")
    print("   weighted avg = mean weighted by class support)\n")
    # Indent each line of the sklearn report for consistent formatting
    for line in report.splitlines():
        print("  " + line)

    print("\n  Confusion Matrix")
    print("  Rows = Actual label, Columns = Predicted label")
    print("  Label order:", " | ".join(LABEL_ORDER))
    print()

    # Header row
    col_width = 12
    header = "  " + " " * col_width + "".join(
        lbl[:col_width].center(col_width) for lbl in LABEL_ORDER
    )
    print(header)
    print("  " + "-" * (col_width + col_width * len(LABEL_ORDER)))

    for i, row_label in enumerate(LABEL_ORDER):
        row_str = "  " + row_label[:col_width].ljust(col_width)
        for val in cm[i]:
            row_str += str(val).center(col_width)
        print(row_str)

    print()

    # ------------------------------------------------------------------
    # 6. Save artifacts
    # ------------------------------------------------------------------
    print("[6/6] Saving model artifacts …")

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"  Saved vectorizer  → {VECTORIZER_PATH}")

    joblib.dump(classifier, CLASSIFIER_PATH)
    print(f"  Saved classifier  → {CLASSIFIER_PATH}")

    # Save the label list in the same order as the classifier's classes.
    # Using classifier.classes_ (not LABEL_ORDER) as the authoritative order
    # so that whatever order sklearn fitted internally is preserved correctly.
    labels_list = list(classifier.classes_)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_list, f, indent=2)
    print(f"  Saved labels      → {LABELS_PATH}")

    print("\n" + "=" * 65)
    print("  Baseline training complete.")
    print("  Review the results above before connecting to FastAPI.")
    print("  test.tsv has NOT been used.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
