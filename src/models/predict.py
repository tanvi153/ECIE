"""
Sentiment Prediction and Inference Utility for ECIE.

Loads trained unified sentiment pipeline (models/sentiment_pipeline.joblib)
or legacy components and predicts customer review polarity with confidence scores
across 3 classes: negative, neutral, positive.
"""

import argparse
from pathlib import Path
from typing import Dict, Union
import joblib

from src.config import (
    SENTIMENT_MODEL_PATH,
    SENTIMENT_PIPELINE_PATH,
    TFIDF_VECTORIZER_PATH,
)
from src.train_sentiment import expand_contractions

_pipeline = None
_model = None
_vectorizer = None


def get_inference_pipeline(pipeline_path: Path = SENTIMENT_PIPELINE_PATH):
    """
    Cached loader for unified sentiment pipeline or legacy decoupled artifacts.
    """
    global _pipeline, _model, _vectorizer

    if _pipeline is not None:
        return _pipeline

    if pipeline_path.exists():
        _pipeline = joblib.load(pipeline_path)
        return _pipeline

    # Fallback to decoupled legacy model & vectorizer
    if _model is None:
        if not SENTIMENT_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Sentiment model not found at '{pipeline_path}' or '{SENTIMENT_MODEL_PATH}'. "
                "Please run 'python -m src.train_sentiment' to train the model."
            )
        _model = joblib.load(SENTIMENT_MODEL_PATH)

    if _vectorizer is None:
        if not TFIDF_VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at '{TFIDF_VECTORIZER_PATH}'. "
                "Please run 'python -m src.train_sentiment' to train the model."
            )
        _vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

    return None


def clean_input_text(text: str) -> str:
    """Preprocesses input text matching training transformations with ordered contraction expansion."""
    if not isinstance(text, str):
        return ""
    return expand_contractions(text.lower().strip())


def predict_sentiment(text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
    """
    Predicts 3-class sentiment (negative, neutral, positive) for a given review text.

    Args:
        text (str): Product review title and/or review body.

    Returns:
        dict: Result containing predicted sentiment, confidence, and class probabilities.
    """
    pipeline = get_inference_pipeline()
    cleaned = clean_input_text(text)

    if pipeline is not None:
        prediction = pipeline.predict([cleaned])[0]
        result = {
            "text": text,
            "sentiment": str(prediction),
        }
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba([cleaned])[0]
            classes = pipeline.classes_
            result["confidence"] = float(max(probs))
            result["probabilities"] = {str(c): float(p) for c, p in zip(classes, probs)}
        elif hasattr(pipeline.named_steps.get("clf"), "predict_proba"):
            clf = pipeline.named_steps["clf"]
            tfidf = pipeline.named_steps["tfidf"]
            features = tfidf.transform([cleaned])
            probs = clf.predict_proba(features)[0]
            classes = clf.classes_
            result["confidence"] = float(max(probs))
            result["probabilities"] = {str(c): float(p) for c, p in zip(classes, probs)}
    else:
        # Legacy fallback
        global _model, _vectorizer
        features = _vectorizer.transform([cleaned])
        prediction = _model.predict(features)[0]
        result = {
            "text": text,
            "sentiment": str(prediction),
        }
        if hasattr(_model, "predict_proba"):
            probs = _model.predict_proba(features)[0]
            classes = _model.classes_
            result["confidence"] = float(max(probs))
            result["probabilities"] = {str(c): float(p) for c, p in zip(classes, probs)}

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment of a customer review")
    parser.add_argument("text", type=str, help="Review text to classify")
    args = parser.parse_args()

    result = predict_sentiment(args.text)
    print("\n--- Sentiment Prediction ---")
    print(f"Review:     {result['text']}")
    print(f"Sentiment:  {result['sentiment'].upper()}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print(f"Probabilities: {result['probabilities']}")


if __name__ == "__main__":
    main()
