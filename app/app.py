import os
import re
import string
import joblib
import time
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ------------------------------------------------------------------
# Config — kept as constants / env overrides so paths are easy to
# swap for deployment (e.g. artifacts pulled from S3 into a mounted
# volume, rather than shipped inside the Docker image).
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    BASE_DIR / "artifacts" / "traditional_model_trainer" / "best_model.pkl",
))
VECTORIZER_PATH = Path(os.getenv(
    "VECTORIZER_PATH",
    BASE_DIR / "artifacts" / "data_transformation" / "vectorizers" / "tfidf_vectorizer.pkl",
))

# AG News label convention confirmed from data validation logs:
# {1: World, 2: Sports, 3: Business, 4: Sci/Tech}
LABEL_MAP = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

CATEGORY_META = {
    "World":    {"code": "WORLD", "color": "#4E7392", "icon": "🌍"},
    "Sports":   {"code": "SPORT", "color": "#4F8F6B", "icon": "⚽"},
    "Business": {"code": "BIZ",   "color": "#BB8F3C", "icon": "💼"},
    "Sci/Tech": {"code": "TECH",  "color": "#3E9494", "icon": "🔬"},
}

# ------------------------------------------------------------------
# Personal / project info — centralized so the About page can be
# edited in one place without touching template markup. Mirrors the
# content.js pattern used in the portfolio site.
# ------------------------------------------------------------------
PERSON_INFO = {
    "name": "Sagar Rai",
    "role": "Machine Learning Engineer · MLOps · Cloud Deployment",
    "location": "Bihar, India",
    "email": "sagarrai9547@gmail.com",
    "phone": "", #//phn number
    "linkedin": "https://linkedin.com/in/mr-raii",
    "github": "https://github.com/mr-raii",
    "bio": (
        "I'm a Data Science graduate building production-style ML systems end to end, "
        "not just notebooks. This wire desk is one project in a portfolio built around one "
        "consistent MLOps identity: config-driven pipelines, experiment tracking, "
        "containerization, and CI/CD to the cloud, applied the same way across every project."
    ),
}

PROJECT_INFO = {
    "title": "The Fourth Wire",
    "subtitle": "AG News Classification Engine",
    "description": (
        "A 4-way news category classifier trained on the AG News corpus — 120,000 labeled "
        "headlines split across World, Sports, Business, and Sci/Tech. Built as a modular, "
        "config-driven pipeline (ingestion, validation, transformation, training, evaluation) "
        "rather than a single notebook, with four candidate models benchmarked head-to-head "
        "before a winner was selected on validation performance."
    ),
    "dataset": "AG News · 120,000 records · 4 balanced classes (30,000 each)",
    "pipeline": [
        "Data Ingestion — sourced from Hugging Face, cached locally to skip re-downloads",
        "Data Validation — schema check, label-range check, null & duplicate detection",
        "Data Transformation — title + description merged, NLTK preprocessing "
        "(tokenize → stopword removal → lemmatization), TF-IDF vectorization "
        "(30,000 features, unigrams + bigrams)",
        "Model Training — RandomizedSearchCV with StratifiedKFold CV across 4 candidates, "
        "config-driven hyperparameter spaces (no hardcoded params)",
        "Model Evaluation — held out test set, classification report, confusion matrix",
    ],
    "models_compared": [
        {"name": "LinearSVC", "macro_f1": 0.9235, "note": "Selected — best validation Macro F1"},
        {"name": "Logistic Regression", "macro_f1": 0.9189, "note": "Close second, cheaper to train"},
        {"name": "Multinomial Naive Bayes", "macro_f1": 0.9087, "note": "Fastest — under 2 seconds"},
        {"name": "LightGBM", "macro_f1": 0.8957, "note": "51 min to train, worst result on sparse TF-IDF"},
    ],
    "test_metrics": {
        "accuracy": 0.9228,
        "macro_f1": 0.9226,
        "weighted_f1": 0.9226,
    },
    "stack": [
        "Python", "Scikit-Learn", "NLTK", "TF-IDF", "LinearSVC",
        "Flask", "MLflow", "DVC", "Docker", "GitHub Actions", "AWS",
    ],
    "status": "Deployment planned — training and prediction pipeline are live; "
              "containerization and cloud deploy are next.",
    "model_name": "LinearSVC",
    "vectorizer_desc": "TF-IDF (30,000 features)",
}

app = Flask(__name__)


# ------------------------------------------------------------------
# NLTK resources — download once if missing. Mirrors the project's
# own nltk_utils pattern: try to find the resource, download only
# on a miss, so repeat runs don't re-download anything.
# ------------------------------------------------------------------
def _ensure_nltk_resources():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for lookup_path, package in resources.items():
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_resources()

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def preprocess_text(raw_text: str) -> str:
    """
    Mirrors the training-time TextPreprocessor so inference sees the
    same distribution the vectorizer was fit on: lowercase, strip
    punctuation and digits, tokenize, drop stopwords, lemmatize.
    """
    text = raw_text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\d+", " ", text)
    tokens = word_tokenize(text)
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in STOPWORDS
    ]
    return " ".join(cleaned_tokens)


# ------------------------------------------------------------------
# Model + vectorizer — loaded once at startup. If artifacts aren't
# present (e.g. fresh clone before training has run), the app still
# boots so the UI and About page are viewable; /predict returns a
# clear error instead of crashing the process.
# ------------------------------------------------------------------
_model = None
_vectorizer = None
_load_error = None


def _load_artifacts():
    global _model, _vectorizer, _load_error
    try:
        with open(MODEL_PATH, "rb") as f:
            _model = joblib.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            _vectorizer = joblib.load(f)
    except FileNotFoundError as exc:
        _load_error = (
            f"Model artifacts not found ({exc}). "
            "Run the training pipeline first, or check MODEL_PATH / VECTORIZER_PATH."
        )


_load_artifacts()


def _confidence_scores(decision_values: np.ndarray, classes: np.ndarray) -> dict:
    """
    LinearSVC has no predict_proba. This applies softmax over the
    one-vs-rest decision_function margins purely to render a
    confidence bar in the UI — it is NOT a calibrated probability.
    """
    shifted = decision_values - np.max(decision_values)
    exp_scores = np.exp(shifted)
    softmax = exp_scores / exp_scores.sum()
    return {
        LABEL_MAP.get(int(cls), str(cls)): round(float(score) * 100, 1)
        for cls, score in zip(classes, softmax)
    }


@app.route("/")
def index():
    return render_template("index.html", categories=CATEGORY_META, project=PROJECT_INFO)


@app.route("/about")
def about():
    return render_template("about.html", person=PERSON_INFO, project=PROJECT_INFO)


@app.route("/predict", methods=["POST"])
def predict():
    if _model is None or _vectorizer is None:
        return jsonify({"error": _load_error or "Model is not loaded."}), 503

    payload = request.get_json(silent=True) or {}
    raw_text = (payload.get("text") or "").strip()

    if not raw_text:
        return jsonify({"error": "Paste a headline or article snippet first."}), 400
    if len(raw_text) < 10:
        return jsonify({"error": "Give it a little more text — a few words won't classify well."}), 400

    start_time = time.perf_counter()

    cleaned = preprocess_text(raw_text)
    if not cleaned:
        return jsonify({"error": "That text left nothing usable after cleaning — try a real headline."}), 400

    vector = _vectorizer.transform([cleaned])
    predicted_label = int(_model.predict(vector)[0])
    category = LABEL_MAP.get(predicted_label, "Unknown")
    meta = CATEGORY_META.get(category, {"code": "?", "color": "#888888", "icon": "📰"})

    confidences = {}
    if hasattr(_model, "decision_function"):
        decision_values = _model.decision_function(vector)[0]
        confidences = _confidence_scores(decision_values, _model.classes_)

    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    return jsonify({
        "category": category,
        "code": meta["code"],
        "color": meta["color"],
        "icon": meta.get("icon", "📰"),
        "confidences": confidences,
        "latency_ms": latency_ms,
    })


@app.route("/healthz")
def healthz():
    """Lightweight health check for load balancers / EB / EC2."""
    ok = _model is not None and _vectorizer is not None
    return jsonify({"status": "ok" if ok else "degraded", "error": _load_error}), (200 if ok else 503)


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
