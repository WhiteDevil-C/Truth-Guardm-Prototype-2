"""
TruthGuard backend test suite.

Run from the project root (with the venv activated and dependencies
installed):

    pytest backend/tests -v

These tests exercise the real, saved model artifacts in backend/model/.
They do not mock the classifier, so they also double as a sanity check that
the vectorizer, classifier, and labels.json are mutually consistent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)

EXPECTED_CLASSES = {
    "true", "mostly-true", "half-true", "barely-true", "false", "pants-fire",
}


def test_model_loaded_at_import_time():
    """The model artifacts must load without raising, and classes must match
    the six LIAR labels the API contract promises."""
    assert app_module.model_error is None
    assert app_module.vectorizer is not None
    assert app_module.classifier is not None
    assert set(app_module.MODEL_CLASSES) == EXPECTED_CLASSES


def test_health_endpoint_reports_model_loaded():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert set(body["classes"]) == EXPECTED_CLASSES
    assert body["model_error"] is None


def test_health_alias_matches_root():
    assert client.get("/health").json()["status"] == "ok"


def test_empty_string_is_handled_safely():
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "uncertain"
    assert body["confidence"] == 0.0
    assert body["class_probabilities"] == {}


def test_whitespace_only_is_handled_safely():
    response = client.post("/analyze", json={"text": "   \n\t  "})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "uncertain"


def test_missing_field_is_treated_as_empty_text():
    """A missing `text` field is treated the same as empty text (graceful
    'uncertain' response) rather than a hard validation error, so the
    extension never has to special-case this condition."""
    response = client.post("/analyze", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "uncertain"


def test_wrong_type_for_text_returns_422():
    response = client.post("/analyze", json={"text": 12345})
    assert response.status_code == 422


def test_oversized_text_rejected_with_422():
    huge_text = "a" * 200_001
    response = client.post("/analyze", json={"text": huge_text})
    assert response.status_code == 422


def test_normal_claim_returns_valid_prediction():
    response = client.post(
        "/analyze",
        json={"text": "The unemployment rate dropped to a 50-year low this year."},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["label"] in EXPECTED_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["word_count"] > 0
    assert "caution" in body and len(body["caution"]) > 0


def test_class_probabilities_are_finite_nonnegative_and_sum_to_one():
    response = client.post(
        "/analyze",
        json={"text": "Crime rates have fallen significantly since the new policy began."},
    )
    body = response.json()
    probs = body["class_probabilities"]

    assert set(probs.keys()) == EXPECTED_CLASSES
    total = 0.0
    for value in probs.values():
        assert value == value  # not NaN
        assert value >= 0.0
        total += value
    assert abs(total - 1.0) < 0.01


def test_explanation_fields_present_and_well_formed():
    response = client.post(
        "/analyze",
        json={"text": "The senator voted against the new healthcare bill twice."},
    )
    body = response.json()
    explanation = body["explanation"]

    assert "available" in explanation
    assert "positive_clues" in explanation
    assert "negative_clues" in explanation
    assert "note" in explanation

    if explanation["available"]:
        for clue in explanation["positive_clues"] + explanation["negative_clues"]:
            assert "term" in clue and isinstance(clue["term"], str)
            assert "contribution" in clue and isinstance(clue["contribution"], float)


def test_explanation_terms_are_consistent_with_fitted_model():
    """Every term returned in an explanation must actually be part of the
    fitted vectorizer's vocabulary, proving the explanation is read directly
    off the real model rather than fabricated."""
    text = "The governor raised taxes on small businesses last year."
    response = client.post("/analyze", json={"text": text})
    explanation = response.json()["explanation"]

    if not explanation["available"]:
        return

    vocabulary = set(app_module.vectorizer.get_feature_names_out())
    for clue in explanation["positive_clues"] + explanation["negative_clues"]:
        assert clue["term"] in vocabulary


def test_out_of_vocabulary_text_reports_unavailable_explanation():
    response = client.post("/analyze", json={"text": "zzxqxz qzjklpvw asdfghqwerty"})
    body = response.json()
    assert body["explanation"]["available"] is False


def test_long_text_triggers_input_warning():
    long_text = "The economy improved steadily this year. " * 20
    response = client.post("/analyze", json={"text": long_text})
    body = response.json()
    assert body["word_count"] > 100
    assert body["input_warning"] is not None


def test_short_claim_has_no_input_warning():
    response = client.post("/analyze", json={"text": "Taxes went up this year."})
    body = response.json()
    assert body["input_warning"] is None


def test_evidence_endpoint_returns_search_links_without_fabricating_results():
    response = client.post("/evidence", json={"text": "The moon landing was staged."})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert len(body["sources"]) > 0
    for source in body["sources"]:
        assert source["url"].startswith("https://")
        assert "name" in source
    # The note must make clear these are not automated verification results.
    assert "not automated" in body["note"].lower()


def test_evidence_endpoint_handles_empty_text():
    response = client.post("/evidence", json={"text": ""})
    body = response.json()
    assert body["available"] is False
    assert body["sources"] == []


def test_missing_model_files_produce_a_useful_error(tmp_path, monkeypatch):
    """Simulate missing/corrupt model artifacts and confirm the API reports
    a clear error instead of crashing or fabricating a prediction."""
    monkeypatch.setattr(app_module, "classifier", None)
    monkeypatch.setattr(app_module, "vectorizer", None)
    monkeypatch.setattr(app_module, "model_error", "Simulated missing artifact for test")

    response = client.post("/analyze", json={"text": "Some claim to analyse."})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "unavailable"
    assert "Simulated missing artifact" in body["caution"]

    health_response = client.get("/")
    assert health_response.json()["model_loaded"] is False
