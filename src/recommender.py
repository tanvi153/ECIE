"""
Enterprise E-Commerce Intelligence Engine (E-CIE)
Content-Based Product Recommendation Engine.

Computes cosine similarity over TF-IDF feature vectors of product metadata
(product_title + category + main_category) to provide instant recommendations
by product ASIN or free-text search queries.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Union
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.config import PRODUCTS_SAMPLE_CSV_PATH, RECOMMENDER_MODEL_PATH


class RecommenderEngine:
    """
    In-memory content-based recommendation engine powered by TF-IDF and NearestNeighbors.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = RECOMMENDER_MODEL_PATH,
        data_path: Union[str, Path] = PRODUCTS_SAMPLE_CSV_PATH,
    ):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.nn_model: Optional[NearestNeighbors] = None
        self.products_df: Optional[pd.DataFrame] = None
        self._asin_to_idx: Dict[str, int] = {}
        self.load()

    def load(self) -> None:
        """
        Loads the trained model artifact and product catalog metadata.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Recommender model artifact not found at {self.model_path}. "
                "Run `python src/train_recommender.py` first."
            )

        artifact = joblib.load(self.model_path)
        self.vectorizer = artifact["vectorizer"]
        self.nn_model = artifact["nn_model"]

        # If artifact contains pre-packaged products dataframe, prefer it; else load CSV
        if "products_df" in artifact and artifact["products_df"] is not None:
            self.products_df = artifact["products_df"].copy()
        elif self.data_path.exists():
            self.products_df = pd.read_csv(self.data_path)
        else:
            raise FileNotFoundError(
                f"Products catalog not found in artifact or at {self.data_path}."
            )

        # Build fast O(1) ASIN lookup index
        self._asin_to_idx = {
            str(asin).strip().upper(): idx
            for idx, asin in enumerate(self.products_df["parent_asin"])
        }

    def recommend(self, query_or_asin: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        Returns top-N recommended products for a given ASIN or search query.

        Args:
            query_or_asin (str): An Amazon parent_asin (e.g. 'B07N1572VL') or free-text query.
            n (int): Number of recommendations to return (default: 5).

        Returns:
            List[Dict[str, Any]]: List of recommended product records with similarity scores.
        """
        if self.products_df is None or self.vectorizer is None or self.nn_model is None:
            raise RuntimeError("RecommenderEngine is not loaded properly.")

        query_str = str(query_or_asin).strip()
        if not query_str or n <= 0:
            return []

        upper_query = query_str.upper()

        # Branch 1: Query matches an exact parent_asin in the catalog
        if upper_query in self._asin_to_idx:
            target_idx = self._asin_to_idx[upper_query]
            item_vec = self.nn_model._fit_X[target_idx]
            # Fetch n + 1 neighbors to exclude the target item itself
            distances, indices = self.nn_model.kneighbors(item_vec, n_neighbors=n + 1)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == target_idx:
                    continue
                row = self.products_df.iloc[idx]
                sim_score = round(max(0.0, 1.0 - float(dist)), 4)
                results.append({
                    "parent_asin": str(row["parent_asin"]),
                    "title": str(row["product_title"]),
                    "product_title": str(row["product_title"]),
                    "category": str(row.get("category", "")),
                    "main_category": str(row.get("main_category", "")),
                    "price": float(row["price"]) if pd.notnull(row.get("price")) else None,
                    "avg_rating": float(row["avg_rating"]) if pd.notnull(row.get("avg_rating")) else None,
                    "review_count": int(row["review_count"]) if pd.notnull(row.get("review_count")) else 0,
                    "similarity_score": sim_score,
                })
                if len(results) == n:
                    break
            return results

        # Branch 2: Query is treated as free-text search
        q_vec = self.vectorizer.transform([query_str])
        if q_vec.nnz == 0:
            # Query terms are completely out of vocabulary
            return []

        fetch_count = min(n, len(self.products_df))
        distances, indices = self.nn_model.kneighbors(q_vec, n_neighbors=fetch_count)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            row = self.products_df.iloc[idx]
            sim_score = round(max(0.0, 1.0 - float(dist)), 4)
            results.append({
                "parent_asin": str(row["parent_asin"]),
                "title": str(row["product_title"]),
                "product_title": str(row["product_title"]),
                "category": str(row.get("category", "")),
                "main_category": str(row.get("main_category", "")),
                "price": float(row["price"]) if pd.notnull(row.get("price")) else None,
                "avg_rating": float(row["avg_rating"]) if pd.notnull(row.get("avg_rating")) else None,
                "review_count": int(row["review_count"]) if pd.notnull(row.get("review_count")) else 0,
                "similarity_score": sim_score,
            })

        return results


# Global singleton instance for easy import and cached usage
_global_engine: Optional[RecommenderEngine] = None


def get_recommender(
    model_path: Union[str, Path] = RECOMMENDER_MODEL_PATH,
    data_path: Union[str, Path] = PRODUCTS_SAMPLE_CSV_PATH,
) -> RecommenderEngine:
    """
    Returns a shared singleton instance of RecommenderEngine.
    """
    global _global_engine
    if _global_engine is None:
        _global_engine = RecommenderEngine(model_path=model_path, data_path=data_path)
    return _global_engine


def recommend(query_or_asin: str, n: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function providing `recommend(query_or_asin, n=5)` as required.

    Args:
        query_or_asin (str): Product ASIN or natural language query.
        n (int): Number of similar products to return.

    Returns:
        List[Dict[str, Any]]: List containing title, category, price, and similarity score.
    """
    engine = get_recommender()
    return engine.recommend(query_or_asin, n=n)
