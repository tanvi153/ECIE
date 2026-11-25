"""
Test Suite for ECIE Content-Based Product Recommendation Engine.
"""

from pathlib import Path
import pytest

from src.config import PRODUCTS_SAMPLE_CSV_PATH, RECOMMENDER_MODEL_PATH
from src.recommender import RecommenderEngine, get_recommender, recommend


def test_recommender_artifacts_exist():
    """Validates that recommender model and catalog sample exist."""
    assert RECOMMENDER_MODEL_PATH.exists(), f"Missing {RECOMMENDER_MODEL_PATH}"
    assert PRODUCTS_SAMPLE_CSV_PATH.exists(), f"Missing {PRODUCTS_SAMPLE_CSV_PATH}"


def test_recommender_engine_initialization():
    """Validates engine initialization and catalog loading."""
    engine = get_recommender()
    assert engine.vectorizer is not None, "Vectorizer was not loaded"
    assert engine.nn_model is not None, "NearestNeighbors model was not loaded"
    assert engine.products_df is not None, "Catalog DataFrame was not loaded"
    assert len(engine.products_df) > 0, "Catalog DataFrame is empty"


def test_recommend_by_asin_returns_n_results():
    """Validates that querying a known ASIN returns exactly n results excluding itself."""
    engine = get_recommender()
    sample_asin = engine.products_df.iloc[0]["parent_asin"]

    n = 5
    recs = recommend(sample_asin, n=n)
    assert len(recs) == n, f"Expected {n} recommendations, got {len(recs)}"

    for item in recs:
        assert item["parent_asin"] != sample_asin, "Target product was returned as recommendation"
        assert "title" in item and len(item["title"]) > 0
        assert "category" in item
        assert "price" in item
        assert "similarity_score" in item
        assert 0.0 <= item["similarity_score"] <= 1.0


def test_recommend_free_text_query():
    """Validates recommendations from natural language search."""
    recs = recommend("Countertop Ice Maker", n=3)
    assert len(recs) == 3
    for r in recs:
        assert "similarity_score" in r
        assert r["similarity_score"] > 0.0
        assert "Ice" in r["title"] or "Maker" in r["title"] or "Appliances" in r["category"]


def test_recommend_unknown_asin_handled_gracefully():
    """Validates that non-existent ASIN does not crash and returns empty list or fallback."""
    unknown_asin = "NONEXISTENT_ASIN_99999"
    recs = recommend(unknown_asin, n=5)
    assert isinstance(recs, list)
    assert len(recs) == 0


def test_recommend_empty_string_handled_gracefully():
    """Validates that empty or whitespace query returns empty list without error."""
    assert recommend("", n=5) == []
    assert recommend("   ", n=5) == []
    assert recommend("", n=0) == []
    assert recommend("ice maker", n=0) == []
