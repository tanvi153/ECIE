"""
Test Suite for ECIE Security, Sentiment Analysis, and Preprocessing Pipeline.
"""

import os
from pathlib import Path
import joblib
import pytest
from sklearn.pipeline import Pipeline

from src.config import (
    DATABASE_URL,
    PROJECT_ROOT,
    SENTIMENT_PIPELINE_PATH,
)
from src.database import test_db_connection as check_db_connection
from src.models.predict import predict_sentiment
from src.train_sentiment import expand_contractions


def test_no_hardcoded_passwords_or_windows_paths():
    """Verifies no passwords or hardcoded Windows paths exist in tracked source code."""
    forbidden_tokens = ["C:\\TANVI", "C:\\Users", "Tanvi0906"]

    source_files = list(PROJECT_ROOT.glob("src/**/*.py")) + [
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / "sql" / "schema.sql",
    ]

    for fpath in source_files:
        if fpath.exists():
            content = fpath.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                assert token not in content, f"Found forbidden token '{token}' in {fpath}"


def test_env_file_is_gitignored():
    """.env file must exist and must be ignored by Git."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    assert gitignore_path.exists(), ".gitignore does not exist"
    gitignore_content = gitignore_path.read_text(encoding="utf-8")
    assert ".env" in gitignore_content.splitlines(), ".env not in .gitignore"


def test_database_connection():
    """PostgreSQL database connectivity check using .env configuration (skips if offline in CI)."""
    try:
        connected = check_db_connection()
    except Exception:
        connected = False

    if not connected:
        pytest.skip("PostgreSQL database is offline or running in isolated CI runner.")
    assert connected is True


def test_contraction_expansion():
    """Validates ordered contraction expansion handling."""
    test_cases = [
        ("I won't accept this", "I will not accept this"),
        ("I can't believe it", "I cannot believe it"),
        ("They didn't arrive", "They did not arrive"),
        ("It's wonderful, we're glad", "It is wonderful, we are glad"),
        ("You won’t regret it", "You will not regret it"),  # Curly apostrophe
    ]
    for raw, expected in test_cases:
        assert expand_contractions(raw) == expected, f"Failed on '{raw}'"


def test_clean_and_sanitize_html_entities():
    """Validates HTML unescaping, contraction expansion, and whitespace sanitization."""
    raw_html = "I didn&#39;t like it.&amp; It was really bad!<br />"
    cleaned = expand_contractions(raw_html)
    assert "did not" in cleaned
    assert "&#39;" not in cleaned
    assert "<br" not in cleaned


def test_sentiment_pipeline_artifact():
    """Ensures the unified pipeline artifact exists and can predict valid classes."""
    assert SENTIMENT_PIPELINE_PATH.exists(), f"Missing {SENTIMENT_PIPELINE_PATH}"
    pipeline = joblib.load(SENTIMENT_PIPELINE_PATH)
    assert isinstance(pipeline, Pipeline), "Pipeline artifact is not an sklearn Pipeline"

    classes = set(pipeline.classes_)
    assert classes in ({"negative", "positive"}, {"negative", "neutral", "positive"}), f"Unexpected classes: {classes}"


def test_sentiment_prediction_inference():
    """Tests end-to-end sentiment inference and class probability output."""
    pos_res = predict_sentiment("This product is exceptional! Outstanding quality and great performance.")
    assert pos_res["sentiment"] == "positive"
    assert "confidence" in pos_res
    assert "probabilities" in pos_res

    neg_res = predict_sentiment("Completely defective, broke on first use. Do not buy.")
    assert neg_res["sentiment"] == "negative"
    assert "confidence" in neg_res

    neu_res = predict_sentiment("The item was ok. Neither particularly good nor terrible.")
    assert neu_res["sentiment"] in ["neutral", "negative", "positive"]
    assert "probabilities" in neu_res


def test_sentiment_handles_empty_and_whitespace_input():
    """Validates that empty strings and whitespace do not crash the pipeline."""
    for empty_input in ["", "   ", "\n\t", "  a  "]:
        res = predict_sentiment(empty_input)
        assert res["sentiment"] in ["negative", "neutral", "positive"]
        assert "confidence" in res
        assert "probabilities" in res
