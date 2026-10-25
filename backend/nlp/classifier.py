"""
STAGE 2 — Complaint Classification
Day 2 of the build plan.

Trains a simple, fast TF-IDF + Logistic Regression classifier on
data/complaints.csv. This is intentionally not a deep model: for a
2-week MVP, a classical model that is fast to train, easy to explain,
and easy to evaluate (precision/recall/F1) is the right choice.

Run this file directly to train and save the model:
    python backend/nlp/classifier.py
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier_model.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "complaints.csv")


def train_classifier(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    """Train the classifier and persist it to disk with joblib."""
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "category"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.25, random_state=42, stratify=df["category"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation — this is what goes into your Day 13 report.
    y_pred = pipeline.predict(X_test)
    print("=== Classification report (held-out test set) ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")
    return pipeline


_model = None


def load_classifier(model_path: str = MODEL_PATH):
    """Load the trained model, training it on first run if missing."""
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(model_path):
        print("No trained model found — training now...")
        _model = train_classifier()
    else:
        _model = joblib.load(model_path)
    return _model


def classify_complaint(text: str) -> dict:
    """
    Predict the category of a single complaint.
    Returns {"category": str, "confidence": float}
    """
    model = load_classifier()
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    best_idx = proba.argmax()
    return {
        "category": classes[best_idx],
        "confidence": round(float(proba[best_idx]), 3),
    }


if __name__ == "__main__":
    train_classifier()
    sample = "There has been no water supply in MVP Colony for four days."
    print(classify_complaint(sample))
