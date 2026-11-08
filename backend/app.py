from pathlib import Path
from urllib.parse import quote_plus
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

BACKEND_DIR = Path(__file__).resolve().parent
PRODUCTION_MODEL_DIR = BACKEND_DIR / "model"
SIGMOID_MODEL_DIR = (
    BACKEND_DIR / "experiments" / "tuned_model" / "sigmoid_candidate"
)

MODEL_MODE = os.getenv("TRUTHGUARD_MODEL", "production").strip().lower()

if MODEL_MODE not in {"production", "sigmoid"}:
    raise ValueError("TRUTHGUARD_MODEL must be 'production' or 'sigmoid'.")

TOP_K_CLUES = 8

app = FastAPI(title="TruthGuard API", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorizer = None
classifier = None
xai_classifier = None
labels = []
model_error = None

try:
    if MODEL_MODE == "sigmoid":
        selected_dir = SIGMOID_MODEL_DIR
        classifier_path = selected_dir / "calibrated_classifier.pkl"
        xai_path = selected_dir / "xai_classifier.pkl"
    else:
        selected_dir = PRODUCTION_MODEL_DIR
        classifier_path = selected_dir / "classifier.pkl"
        xai_path = classifier_path

    vectorizer = joblib.load(selected_dir / "vectorizer.pkl")
    classifier = joblib.load(classifier_path)
    xai_classifier = joblib.load(xai_path)

    with open(selected_dir / "labels.json", "r", encoding="utf-8") as fh:
        labels = json.load(fh)

    MODEL_CLASSES = [str(c) for c in classifier.classes_]

    if MODEL_CLASSES != [str(c) for c in xai_classifier.classes_]:
        raise ValueError("Prediction and XAI model class ordering does not match.")

    if not hasattr(xai_classifier, "coef_"):
        raise ValueError("XAI classifier does not expose linear coefficients.")

except Exception as exc:
    model_error = f"{type(exc).__name__}: {exc}"
    vectorizer = None
    classifier = None
    xai_classifier = None
    labels = []
    MODEL_CLASSES = []


# Human-readable descriptions shown by the extension next to the raw label.
# These are plain-language glosses of the original LIAR annotation scheme,
# not an independent judgement by TruthGuard.
LABEL_DESCRIPTIONS = {
    "true": "Rated TRUE by the model — statement appears accurate.",
    "mostly-true": "Rated MOSTLY TRUE — accurate but needs clarification or context.",
    "half-true": "Rated HALF TRUE — partially accurate, leaves out important details.",
    "barely-true": "Rated BARELY TRUE — contains an element of truth but is misleading.",
    "false": "Rated FALSE — the model predicts this statement is not accurate.",
    "pants-fire": "Rated PANTS ON FIRE — the model predicts an especially false claim.",
}


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class AnalyzeRequest(BaseModel):
    text: str = Field(
        default="",
        # No min_length here on purpose: both "" and whitespace-only input
        # are routed through the same /analyze logic below so the extension
        # always gets one consistent JSON shape instead of a 422 for "" and
        # a 200 for "   ". The max_length bound exists only to reject
        # abusive payloads.
        max_length=200000,
        description="Claim or page text to analyse.",
    )


class EvidenceRequest(BaseModel):
    text: str = Field(default="", max_length=2000)


# --------------------------------------------------------------------------- #
# Explanation logic
# --------------------------------------------------------------------------- #


def build_explanation(text: str, class_index: int, top_k: int = TOP_K_CLUES) -> dict:
    """
    Return the strongest positive and negative feature contributions to the
    linear score of `class_index`.

    Only features actually present in `text` are considered, so every clue
    listed is a real term from the input. This is a direct read-out of the
    fitted TF-IDF + Logistic Regression model (contribution = tfidf_weight *
    class_coefficient for each vocabulary term found in the text), not a
    post-hoc approximation such as LIME or SHAP. It is only mathematically
    valid for this linear model; if the model is ever replaced with a
    non-linear one, this function must be replaced too.
    """
    if vectorizer is None or classifier is None or xai_classifier is None:
        return {
            "available": False,
            "positive_clues": [],
            "negative_clues": [],
            "note": "Model not loaded, so no explanation is available.",
        }

    try:
        matrix = vectorizer.transform([text])
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "positive_clues": [],
            "negative_clues": [],
            "note": f"Explanation could not be computed: {type(exc).__name__}.",
        }

    if matrix.nnz == 0:
        return {
            "available": False,
            "positive_clues": [],
            "negative_clues": [],
            "note": (
                "None of the words in this text appear in the model's training "
                "vocabulary, so no feature-level explanation could be produced."
            ),
        }

    vocabulary = vectorizer.get_feature_names_out()
    coefficients = xai_classifier.coef_[class_index]

    contributions = []
    for feature_index, tfidf_value in zip(matrix.indices, matrix.data):
        contribution = float(tfidf_value) * float(coefficients[feature_index])
        contributions.append((str(vocabulary[feature_index]), contribution))

    contributions.sort(key=lambda pair: pair[1], reverse=True)

    positive_clues = [
        {"term": term, "contribution": round(value, 5)}
        for term, value in contributions
        if value > 0
    ][:top_k]

    negative_clues = [
        {"term": term, "contribution": round(value, 5)}
        for term, value in reversed(contributions)
        if value < 0
    ][:top_k]

    return {
        "available": True,
        "predicted_class": MODEL_CLASSES[class_index],
        "positive_clues": positive_clues,
        "negative_clues": negative_clues,
        "note": (
            "These are model feature contributions (TF-IDF weight x learned "
            "class coefficient) for words found in your text, not verified "
            "evidence. They show which words raised or lowered the model's "
            "score for this class, and say nothing about whether the claim "
            "is factually true."
        ),
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/")
def health():
    """Health / status endpoint used by the extension's connection indicator."""
    return {
        "status": "ok",
        "service": "TruthGuard API",
        "version": app.version,
        "model_loaded": classifier is not None and vectorizer is not None,
        "classes": labels or MODEL_CLASSES,
        "model_error": model_error,
    }


@app.get("/health")
def health_alias():
    """Alias of / for clients that expect a conventional /health path."""
    return health()


def _empty_response(original_text: str) -> dict:
    return {
        "claim": original_text,
        "label": "uncertain",
        "label_description": "No text was provided, so the model made no prediction.",
        "confidence": 0.0,
        "class_probabilities": {},
        "explanation": {
            "available": False,
            "positive_clues": [],
            "negative_clues": [],
            "note": "No text was provided to analyse.",
        },
        "word_count": 0,
        "input_warning": None,
        "caution": "No text was provided, so no analysis was performed.",
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    text = (request.text or "").strip()

    if not text:
        return _empty_response(request.text)

    if classifier is None or vectorizer is None:
        return {
            "claim": text,
            "label": "unavailable",
            "label_description": "The model is not loaded on the server.",
            "confidence": 0.0,
            "class_probabilities": {},
            "explanation": {
                "available": False,
                "positive_clues": [],
                "negative_clues": [],
                "note": "The trained model could not be loaded on the server.",
            },
            "word_count": len(text.split()),
            "input_warning": None,
            "caution": f"Model unavailable: {model_error}",
        }

    try:
        matrix = vectorizer.transform([text])
        probabilities = classifier.predict_proba(matrix)[0]
        class_index = int(probabilities.argmax())
    except Exception as exc:  # noqa: BLE001 - never turn an inference crash into a fake result
        return {
            "claim": text,
            "label": "error",
            "label_description": "Inference failed.",
            "confidence": 0.0,
            "class_probabilities": {},
            "explanation": {
                "available": False,
                "positive_clues": [],
                "negative_clues": [],
                "note": "No explanation available because inference failed.",
            },
            "word_count": len(text.split()),
            "input_warning": None,
            "caution": f"Inference failed: {type(exc).__name__}: {exc}",
        }

    predicted_label = MODEL_CLASSES[class_index]

    class_probabilities = {
        label: round(float(probabilities[i]), 4)
        for i, label in enumerate(MODEL_CLASSES)
    }

    # The model was trained on short claims (~25 words). Longer text is far
    # outside that distribution, so say so plainly instead of silently
    # returning a confident-looking number.
    word_count = len(text.split())
    input_warning = None
    if word_count > 100:
        input_warning = (
            f"This text is about {word_count} words long. The model was trained on "
            "short claims of roughly 25 words, so a prediction on text this long is "
            "outside its intended use and should not be read as a fact-check."
        )

    return {
        "claim": text,
        "label": predicted_label,
        "label_description": LABEL_DESCRIPTIONS.get(predicted_label, predicted_label),
        "confidence": round(float(probabilities[class_index]), 4),
        "class_probabilities": class_probabilities,
        "explanation": build_explanation(text, class_index),
        "word_count": word_count,
        "input_warning": input_warning,
        "caution": (
            "This is a machine-learning prediction from a six-class model trained "
            "on short political claims. The confidence value is NOT the probability "
            "that the text is true or false, and this output is not a fact-check."
        ),
    }


def _google_fact_check_api_key():
    """Read the Google API key from backend/.env or the environment."""
    key = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()
    if key:
        return key

    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return ""

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "GOOGLE_FACT_CHECK_API_KEY":
            return value.strip().strip('"').strip("'")
    return ""




def _prepare_evidence_query(text: str) -> str:
    """Create a concise search query from selected text or webpage content."""
    text = (text or "").strip()

    if not text:
        return ""

    words = text.split()

    # Preserve short selected claims.
    if len(words) <= 45:
        return text[:300]

    noise_phrases = (
        "skip to content", "sign in", "log in", "subscribe",
        "newsletter", "advertisement", "advertising",
        "all rights reserved", "read more", "latest news",
        "top stories", "weather", "settings", "help", "menu",
    )

    candidates = []

    for line in text.splitlines():
        line = " ".join(line.split())

        if not line:
            continue

        lowered = line.lower()

        if any(phrase in lowered for phrase in noise_phrases):
            continue

        line_words = line.split()

        if not 6 <= len(line_words) <= 35:
            continue

        alphabetic_words = [
            word for word in line_words
            if any(char.isalpha() for char in word)
        ]

        if len(alphabetic_words) < 5:
            continue

        candidates.append(line)

    if not candidates:
        return " ".join(words[:35])[:300]

    # Use the first meaningful paragraph as a focused evidence-search query.
    return candidates[0][:300]


@app.post("/evidence")
def evidence(request: EvidenceRequest):
    """Retrieve Google fact-check reviews and retain manual search fallbacks."""
    text = (request.text or "").strip()

    if not text:
        return {
            "available": False,
            "query": "",
            "status": "empty",
            "fact_checks": [],
            "sources": [],
            "note": "No text was provided, so no search links were generated.",
        }

    query = _prepare_evidence_query(text)
    encoded = quote_plus(query)

    sources = [
        {
            "name": "Google Fact Check Explorer",
            "url": f"https://toolbox.google.com/factcheck/explorer/search/{encoded}",
        },
        {
            "name": "Google Search",
            "url": f"https://www.google.com/search?q={encoded}",
        },
        {
            "name": "Snopes",
            "url": f"https://www.snopes.com/?s={encoded}",
        },
        {
            "name": "PolitiFact",
            "url": f"https://www.politifact.com/search/?q={encoded}",
        },
    ]

    result = {
        "available": True,
        "query": query,
        "status": "unavailable",
        "fact_checks": [],
        "sources": sources,
    }

    api_key = _google_fact_check_api_key()
    if not api_key:
        result["note"] = (
            "Google automated lookup is unavailable because the API key is not "
            "configured. Manual search links are provided; these are not "
            "automated fact-check results."
        )
        return result

    params = urlencode({
        "query": query,
        "languageCode": "en",
        "pageSize": 10,
        "key": api_key,
    })
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search?" + params
    api_request = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(api_request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        result["note"] = (
            f"Google automated lookup failed with HTTP {exc.code}. "
            "Manual search links are provided; these are not automated "
            "fact-check results."
        )
        return result
    except (URLError, TimeoutError, OSError, ValueError):
        result["note"] = (
            "Google automated lookup could not be completed. Manual search "
            "links are provided; these are not automated fact-check results."
        )
        return result

    for claim in payload.get("claims", []):
        reviews = []
        for review in claim.get("claimReview", []):
            publisher = review.get("publisher") or {}
            reviews.append({
                "publisher": publisher.get("name", "Unknown publisher"),
                "publisher_site": publisher.get("site", ""),
                "title": review.get("title", ""),
                "url": review.get("url", ""),
                "review_date": review.get("reviewDate", ""),
                "rating": review.get("textualRating", ""),
                "language_code": review.get("languageCode", ""),
            })

        result["fact_checks"].append({
            "claim_text": claim.get("text", ""),
            "claimant": claim.get("claimant", ""),
            "claim_date": claim.get("claimDate", ""),
            "reviews": reviews,
        })

    if result["fact_checks"]:
        result["status"] = "results"
        result["note"] = (
            "Google returned published fact-check review matches. A match does "
            "not guarantee the review concerns the exact context of your claim; "
            "read the original review before drawing conclusions. Manual search "
            "links are also provided; they are not automated fact-check results."
        )
    else:
        result["status"] = "no_matches"
        result["note"] = (
            "Google returned no matching fact-check reviews for this query. "
            "This does not mean the claim is true or false. Manual search links "
            "are provided; these are not automated fact-check results."
        )

    return result
