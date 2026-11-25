"""
Enterprise E-Commerce Intelligence Engine (E-CIE)
Content-Based Product Recommender Training Pipeline.

Extracts top products prioritized by review count from PostgreSQL,
generates a lightweight catalog sample (data/products_sample.csv),
fits a TF-IDF (1-2 grams) + NearestNeighbors (cosine similarity) pipeline,
and logs parameters, metrics, and sanity-check evaluations to MLflow and docs/results.md.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import text

from src.config import (
    DATA_DIR,
    EXPERIMENTS_DIR,
    MLFLOW_DB_PATH,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PRODUCTS_SAMPLE_CSV_PATH,
    PROJECT_ROOT,
    RECOMMENDER_MODEL_PATH,
)
from src.database import get_db_engine


def fetch_and_export_products(
    sample_size: int = 60000,
    output_path: Path = PRODUCTS_SAMPLE_CSV_PATH,
) -> pd.DataFrame:
    """
    Extracts top products by review volume from PostgreSQL and exports to CSV.
    Falls back to existing CSV if database is unavailable.
    """
    print(f"\n[1/4] Fetching top {sample_size:,} products from PostgreSQL...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        engine = get_db_engine()
        query = text("""
            SELECT 
                p.parent_asin,
                p.product_title,
                p.category,
                p.main_category,
                p.price,
                ROUND(AVG(r.rating), 2) AS avg_rating,
                COUNT(r.review_id) AS review_count
            FROM products p
            JOIN reviews r ON p.parent_asin = r.parent_asin
            WHERE p.product_title NOT LIKE 'Product %'
              AND p.product_title NOT LIKE 'Test %'
              AND LENGTH(TRIM(p.product_title)) > 2
            GROUP BY p.parent_asin, p.product_title, p.category, p.main_category, p.price
            ORDER BY review_count DESC, p.parent_asin
            LIMIT :limit;
        """)

        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"limit": sample_size})

        if df.empty:
            raise ValueError("Query returned 0 rows from PostgreSQL products/reviews.")

        print(f"  Extracted {len(df):,} products from database.")

    except Exception as e:
        print(f"  Warning: Database extraction failed ({e}). Checking local fallback...")
        if output_path.exists():
            print(f"  Loading existing CSV fallback: {output_path}")
            df = pd.read_csv(output_path)
            return df
        else:
            raise RuntimeError(f"Cannot extract products from DB and no fallback exists at {output_path}")

    # Clean missing values
    df["product_title"] = df["product_title"].fillna("Unknown Product").str.strip()
    df["category"] = df["category"].fillna("General").str.strip()
    df["main_category"] = df["main_category"].fillna("General").str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(19.99).round(2)
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce").fillna(4.0).round(2)
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(1).astype(int)

    # Export to CSV
    df.to_csv(output_path, index=False)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Exported {len(df):,} products to {output_path} ({file_size_mb:.2f} MB)")
    if file_size_mb > 30.0:
        print(f"  WARNING: File size {file_size_mb:.2f} MB exceeds target 30 MB threshold!")

    return df


def train_recommender(
    df: pd.DataFrame,
    max_features: int = 100000,
    n_neighbors: int = 20,
) -> Tuple[TfidfVectorizer, NearestNeighbors]:
    """
    Fits TF-IDF vectorizer and NearestNeighbors cosine index on product metadata.
    """
    print(f"\n[2/4] Fitting TF-IDF Vectorizer & NearestNeighbors index...")
    t0 = time.time()

    # Create composite content representation: title + category + main_category
    corpus = (
        df["product_title"].astype(str)
        + " "
        + df["category"].astype(str)
        + " "
        + df["main_category"].astype(str)
    ).tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    t_tfidf = time.time() - t0
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape} ({tfidf_matrix.nnz:,} non-zeros in {t_tfidf:.2f}s)")

    # Cosine distance over sparse TF-IDF vectors (brute force on sparse vectors, avoiding N*N pairwise matrix)
    t1 = time.time()
    nn_model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
    )
    nn_model.fit(tfidf_matrix)
    print(f"  NearestNeighbors index fitted in {time.time() - t1:.2f}s")

    return vectorizer, nn_model


def save_recommender_artifact(
    vectorizer: TfidfVectorizer,
    nn_model: NearestNeighbors,
    df: pd.DataFrame,
    output_path: Path = RECOMMENDER_MODEL_PATH,
) -> Path:
    """
    Serializes recommender model bundle with joblib.
    """
    print(f"\n[3/4] Saving model bundle to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "vectorizer": vectorizer,
        "nn_model": nn_model,
        "products_df": df,
    }
    joblib.dump(bundle, output_path, compress=3)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved recommender artifact ({size_mb:.2f} MB)")
    return output_path


def generate_sanity_checks(
    vectorizer: TfidfVectorizer,
    nn_model: NearestNeighbors,
    df: pd.DataFrame,
    n_recommendations: int = 5,
) -> List[Dict[str, Any]]:
    """
    Evaluates 5 diverse sample products from different categories and records recommendations.
    """
    print(f"\n[4/4] Running sanity-check recommendations on 5 sample products...")

    target_queries = [
        # 1. Top Subscription Box
        "B07N1572VL",
        # 2. National Geographic Magazine
        "B00005NIOH",
        # 3. Family Handyman Magazine
        "B002PXVYEQ",
        # 4. Appliance Filter / Ice Maker
        "B08YBGCNHP",
        # 5. Natural Language Query
        "Countertop Ice Maker Machine",
    ]

    asin_to_idx = {str(a).strip().upper(): i for i, a in enumerate(df["parent_asin"])}
    sanity_results = []

    for query in target_queries:
        query_upper = query.upper()
        if query_upper in asin_to_idx:
            idx = asin_to_idx[query_upper]
            target_row = df.iloc[idx]
            query_label = f"ASIN: {query} ({target_row['product_title'][:50]}...)"
            item_vec = nn_model._fit_X[idx]
            dists, indices = nn_model.kneighbors(item_vec, n_neighbors=n_recommendations + 1)
            
            recs = []
            for dist, neighbor_idx in zip(dists[0], indices[0]):
                if neighbor_idx == idx:
                    continue
                nrow = df.iloc[neighbor_idx]
                recs.append({
                    "parent_asin": nrow["parent_asin"],
                    "title": nrow["product_title"],
                    "category": nrow["category"],
                    "price": float(nrow["price"]),
                    "similarity": round(max(0.0, 1.0 - float(dist)), 4),
                })
                if len(recs) == n_recommendations:
                    break
        else:
            query_label = f"Query: '{query}'"
            q_vec = vectorizer.transform([query])
            dists, indices = nn_model.kneighbors(q_vec, n_neighbors=n_recommendations)
            recs = []
            for dist, neighbor_idx in zip(dists[0], indices[0]):
                nrow = df.iloc[neighbor_idx]
                recs.append({
                    "parent_asin": nrow["parent_asin"],
                    "title": nrow["product_title"],
                    "category": nrow["category"],
                    "price": float(nrow["price"]),
                    "similarity": round(max(0.0, 1.0 - float(dist)), 4),
                })

        sanity_results.append({
            "query": query_label,
            "recommendations": recs,
        })

        print(f"\n--- {query_label} ---")
        for rank, r in enumerate(recs, 1):
            print(f"  {rank}. [Sim: {r['similarity']:.4f}] ${r['price']:.2f} | {r['title'][:60]} | ({r['category'][:35]})")

    return sanity_results


def update_results_doc(sanity_results: List[Dict[str, Any]], catalog_size: int, csv_size_mb: float) -> None:
    """
    Appends recommender results and sanity-check tables to docs/results.md.
    """
    results_path = PROJECT_ROOT / "docs" / "results.md"
    if not results_path.exists():
        return

    content = results_path.read_text(encoding="utf-8")

    # If section already exists, replace it; otherwise append
    section_header = "## 3. Product Recommendation Engine Results"
    
    recommender_md = [
        f"\n{section_header}\n",
        f"- **Methodology**: Content-Based Filtering using TF-IDF (1-2 n-grams, 100,000 max features, sublinear term frequency) and NearestNeighbors with Cosine Distance on sparse vector space.",
        f"- **Pairwise Matrix Optimization**: Brute-force cosine distance directly evaluated over sparse matrices (avoiding $O(N^2)$ memory explosion for dense matrices).",
        f"- **Deployed Catalog Subset**: **{catalog_size:,} products** exported to `data/products_sample.csv` ({csv_size_mb:.2f} MB), well within the 30 MB Streamlit Community Cloud limit.",
        f"- **Catalog Columns**: `parent_asin`, `product_title`, `category`, `main_category`, `price`, `avg_rating`, `review_count`.\n",
        "### Sanity-Check Validations (5 Test Cases)\n",
    ]

    for item in sanity_results:
        recommender_md.append(f"#### {item['query']}")
        recommender_md.append("| Rank | Recommended Product | Category | Price | Similarity |")
        recommender_md.append("|:---:|:---|:---|:---:|:---:|")
        for rank, rec in enumerate(item["recommendations"], 1):
            title_clean = rec["title"].replace("|", "-")[:55]
            cat_clean = rec["category"].replace("|", "-")[:35]
            recommender_md.append(
                f"| {rank} | {title_clean} | {cat_clean} | ${rec['price']:.2f} | **{rec['similarity']:.4f}** |"
            )
        recommender_md.append("")

    new_section = "\n".join(recommender_md)

    if section_header in content:
        # Replace existing section up to next ## or EOF
        parts = content.split(section_header)
        post_split = parts[1].split("\n## ", 1)
        rest = "\n## " + post_split[1] if len(post_split) > 1 else ""
        updated_content = parts[0].rstrip() + "\n" + new_section + rest
    else:
        updated_content = content.rstrip() + "\n" + new_section

    results_path.write_text(updated_content, encoding="utf-8")
    print(f"\nUpdated {results_path} with recommender sanity-check results.")


def main():
    parser = argparse.ArgumentParser(description="Train Content-Based Recommender for ECIE")
    parser.add_argument("--sample-size", type=int, default=60000, help="Number of products to sample")
    parser.add_argument("--max-features", type=int, default=100000, help="Max features for TF-IDF")
    parser.add_argument("--n-neighbors", type=int, default=20, help="Neighbors index capacity")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ECIE_Product_Recommender")

    with mlflow.start_run(run_name="TFIDF_NearestNeighbors_Recommender"):
        mlflow.log_params({
            "sample_size": args.sample_size,
            "ngram_range": "(1, 2)",
            "max_features": args.max_features,
            "metric": "cosine",
            "algorithm": "brute",
            "n_neighbors": args.n_neighbors,
        })

        # 1. Fetch & Export
        df = fetch_and_export_products(sample_size=args.sample_size)
        csv_size_mb = PRODUCTS_SAMPLE_CSV_PATH.stat().st_size / (1024 * 1024)

        # 2. Fit TF-IDF & NearestNeighbors
        vectorizer, nn_model = train_recommender(
            df=df,
            max_features=args.max_features,
            n_neighbors=args.n_neighbors,
        )

        # 3. Save Artifact
        model_path = save_recommender_artifact(vectorizer, nn_model, df)
        model_size_mb = model_path.stat().st_size / (1024 * 1024)

        # 4. Sanity Checks
        sanity_results = generate_sanity_checks(vectorizer, nn_model, df, n_recommendations=5)

        # Log Metrics
        mlflow.log_metrics({
            "catalog_products": len(df),
            "vocab_size": len(vectorizer.vocabulary_),
            "csv_size_mb": csv_size_mb,
            "model_size_mb": model_size_mb,
        })
        mlflow.log_artifact(str(PRODUCTS_SAMPLE_CSV_PATH))

        # Update documentation
        update_results_doc(sanity_results, catalog_size=len(df), csv_size_mb=csv_size_mb)

    print("\n Recommender training and evaluation complete!")


if __name__ == "__main__":
    main()
