"""
Enterprise E-Commerce Intelligence Engine (E-CIE)
Multi-Class Sentiment Model Training Pipeline.

Trains a robust 3-class (negative, neutral, positive) linear sentiment classifier
using a scikit-learn Pipeline (TF-IDF + SGDClassifier/LinearSVC), tracked via MLflow.
Labels are derived strictly from star ratings (1-2: negative, 3: neutral, 4-5: positive)
without keyword filtering or label leakage.
"""

import argparse
import html
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sqlalchemy import text

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROJECT_ROOT,
    SENTIMENT_PIPELINE_PATH,
)
from src.database import get_db_engine

# ==============================================================================
# 1. Contraction Expansion & Text Sanitization Rules (Ordered)
# ==============================================================================
# Specific multi-word / irregular contractions MUST precede single-letter / general 'n't'
# rules to prevent erroneous contractions like 'wo not' or 'ca not'.
ORDERED_CONTRACTIONS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bwon['’]t\b", re.IGNORECASE), "will not"),
    (re.compile(r"\bcan['’]t\b", re.IGNORECASE), "cannot"),
    (re.compile(r"\bshan['’]t\b", re.IGNORECASE), "shall not"),
    (re.compile(r"\blet['’]s\b", re.IGNORECASE), "let us"),
    (re.compile(r"n['’]t\b", re.IGNORECASE), " not"),
    (re.compile(r"['’]re\b", re.IGNORECASE), " are"),
    (re.compile(r"['’]s\b", re.IGNORECASE), " is"),
    (re.compile(r"['’]d\b", re.IGNORECASE), " would"),
    (re.compile(r"['’]ll\b", re.IGNORECASE), " will"),
    (re.compile(r"['’]ve\b", re.IGNORECASE), " have"),
    (re.compile(r"['’]m\b", re.IGNORECASE), " am"),
]

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# Negation and critical polarity words to preserve in TF-IDF vocabulary
SENTIMENT_PRESERVING_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "neither",
    "nobody", "nowhere", "cannot", "without", "against",
    "hardly", "scarcely", "barely",
}

CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS - SENTIMENT_PRESERVING_WORDS)


def expand_contractions(text_value: str) -> str:
    """
    Decodes HTML entities, strips raw HTML tags, and applies ordered regular
    expression rules to expand English contractions. Handles curly apostrophes ('’')
    and prevents 'won't' -> 'wo not' corruption.
    """
    if not isinstance(text_value, str):
        return ""
    # 1. Unescape HTML entities (e.g., didn&#39;t -> didn't, &amp; -> &)
    expanded = html.unescape(text_value)
    # 2. Replace HTML tags (<br />, <p>, etc.) with space
    expanded = HTML_TAG_PATTERN.sub(" ", expanded)
    # 3. Apply ordered contraction expansions
    for pattern, replacement in ORDERED_CONTRACTIONS:
        expanded = pattern.sub(replacement, expanded)
    # 4. Collapse consecutive whitespace
    return MULTI_SPACE_PATTERN.sub(" ", expanded).strip()


def normalize_text_series(series: pd.Series) -> pd.Series:
    """Lowercases, cleans HTML entities/tags, and normalizes contractions across a pandas Series."""
    return series.fillna("").astype(str).str.lower().apply(expand_contractions)


# ==============================================================================
# 2. Database Extraction (Stratified & Balanced)
# ==============================================================================
def fetch_balanced_dataset(
    sample_per_class: int = 100000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Draws a stratified, class-balanced sample from PostgreSQL view 'v_cleaned_reviews'.
    Classes:
      - 'negative' (ratings 1-2)
      - 'neutral'  (rating 3)
      - 'positive' (ratings 4-5)

    Only uses title and text columns. No keyword or heuristic filtering is applied.
    """
    engine = get_db_engine()
    print(f"\n[1/5] Fetching balanced multi-class dataset from PostgreSQL (target: ~{sample_per_class:,} per class)...")

    # Union query fetching up to sample_per_class per sentiment label
    query = text("""
        (
            SELECT 
                review_id,
                COALESCE(review_title, '') AS review_title,
                COALESCE(review_text, '') AS review_text,
                rating,
                sentiment
            FROM v_cleaned_reviews
            WHERE sentiment = 'negative'
              AND (LENGTH(COALESCE(review_title, '')) > 0 OR LENGTH(COALESCE(review_text, '')) > 0)
            LIMIT :sample_size
        )
        UNION ALL
        (
            SELECT 
                review_id,
                COALESCE(review_title, '') AS review_title,
                COALESCE(review_text, '') AS review_text,
                rating,
                sentiment
            FROM v_cleaned_reviews
            WHERE sentiment = 'neutral'
              AND (LENGTH(COALESCE(review_title, '')) > 0 OR LENGTH(COALESCE(review_text, '')) > 0)
            LIMIT :sample_size
        )
        UNION ALL
        (
            SELECT 
                review_id,
                COALESCE(review_title, '') AS review_title,
                COALESCE(review_text, '') AS review_text,
                rating,
                sentiment
            FROM v_cleaned_reviews
            WHERE sentiment = 'positive'
              AND (LENGTH(COALESCE(review_title, '')) > 0 OR LENGTH(COALESCE(review_text, '')) > 0)
            LIMIT :sample_size
        );
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"sample_size": sample_per_class})

    if df.empty:
        raise ValueError(
            "Database view 'v_cleaned_reviews' returned 0 records. "
            "Please ensure reviews and review_text tables are populated."
        )

    counts = df["sentiment"].value_counts().to_dict()
    print(f"      Raw records fetched: {len(df):,} | Breakdown: {counts}")

    # Enforce strict class balance across all 3 classes
    min_count = min(counts.values())
    if min_count < sample_per_class:
        print(f"      [!] Notice: Smallest class has {min_count:,} reviews. Balancing all classes to {min_count:,}.")

    balanced_df = (
        df.groupby("sentiment")
        .sample(n=min_count, random_state=random_state)
        .reset_index(drop=True)
    )

    final_counts = balanced_df["sentiment"].value_counts().to_dict()
    print(f"      Balanced training sample: {len(balanced_df):,} total ({final_counts})")
    return balanced_df


# ==============================================================================
# 3. Model Pipeline Construction
# ==============================================================================
def create_sentiment_pipeline(
    max_features: int = 200000,
    random_state: int = 42,
) -> Pipeline:
    """
    Constructs an end-to-end scikit-learn Pipeline:
    - TfidfVectorizer: 1-2 grams, sublinear TF, sentiment-preserving stop words
    - SGDClassifier: modified_huber loss (supports predict_proba), balanced class weights
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words=CUSTOM_STOP_WORDS,
        strip_accents="unicode",
    )

    classifier = SGDClassifier(
        loss="modified_huber",
        penalty="l2",
        alpha=1e-5,
        max_iter=2000,
        tol=1e-4,
        class_weight="balanced",
        random_state=random_state,
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


# ==============================================================================
# 4. Training, Evaluation, and MLflow Logging
# ==============================================================================
def train_and_evaluate(
    sample_per_class: int = 100000,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 200000,
    model_output_path: Path = SENTIMENT_PIPELINE_PATH,
    cm_output_path: Path = PROJECT_ROOT / "docs" / "confusion_matrix.png",
) -> Dict[str, float]:
    """
    Executes the end-to-end training and evaluation pipeline:
    1. Fetches stratified balanced dataset from PostgreSQL.
    2. Combines review_title + review_text and normalizes contractions.
    3. Splits 80/20 train/test with stratification.
    4. Fits Pipeline(TF-IDF + SGDClassifier).
    5. Evaluates metrics (train/test acc, per-class F1/P/R, macro F1, confusion matrix).
    6. Saves pipeline artifact and confusion matrix PNG.
    7. Logs all parameters, metrics, and artifacts to MLflow.
    """
    # 1. Fetch data
    df = fetch_balanced_dataset(
        sample_per_class=sample_per_class,
        random_state=random_state,
    )

    # 2. Prepare text input (title + text ONLY, never rating or other columns)
    print("\n[2/5] Preparing feature inputs and expanding contractions...")
    raw_text = (df["review_title"].fillna("") + " " + df["review_text"].fillna("")).str.strip()
    X = normalize_text_series(raw_text)
    y = df["sentiment"].astype(str)

    # 3. Train-Test Split (80/20 Stratified)
    print(f"\n[3/5] Splitting dataset into train ({int((1 - test_size) * 100)}%) and test ({int(test_size * 100)}%) with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"      Training samples: {len(X_train):,} | Testing samples: {len(X_test):,}")

    # 4. Build and Train Pipeline
    print(f"\n[4/5] Training unified Pipeline (TF-IDF 1-2 grams [max {max_features:,}] -> SGDClassifier modified_huber)...")
    pipeline = create_sentiment_pipeline(
        max_features=max_features,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)
    print("      Model training complete!")

    # 5. Predictions & Evaluation
    print("\n[5/5] Evaluating performance and calculating metrics...")
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    acc_gap = (train_acc - test_acc) * 100

    target_classes = sorted(list(y.unique()))
    macro_f1 = f1_score(y_test, y_test_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_test_pred, average="weighted")

    precisions = precision_score(y_test, y_test_pred, labels=target_classes, average=None, zero_division=0)
    recalls = recall_score(y_test, y_test_pred, labels=target_classes, average=None, zero_division=0)
    f1s = f1_score(y_test, y_test_pred, labels=target_classes, average=None, zero_division=0)

    cm = confusion_matrix(y_test, y_test_pred, labels=target_classes)

    # Print comprehensive diagnostic report
    print("\n" + "=" * 70)
    print("               ECIE SENTIMENT MODEL EVALUATION REPORT")
    print("=" * 70)
    print(f"Classes:            {target_classes} ({len(target_classes)}-class)")
    print(f"Total Sample Size:  {len(df):,} reviews")
    print(f"Training Accuracy:  {train_acc:.4f} ({train_acc * 100:.2f}%)")
    print(f"Testing Accuracy:   {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Train-Test Gap:     {acc_gap:.2f}% (Overfitting check)")
    print(f"Macro F1-Score:     {macro_f1:.4f}")
    print(f"Weighted F1-Score:  {weighted_f1:.4f}")
    print("\nPer-Class Breakdown:")
    for cls_name, p, r, f in zip(target_classes, precisions, recalls, f1s):
        print(f"  - {cls_name.upper():<9} Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred, digits=4, labels=target_classes, zero_division=0))

    print("\nConfusion Matrix (Rows=True, Columns=Predicted):")
    print(f"Labels: {target_classes}")
    for row_name, row in zip(target_classes, cm):
        print(f"  {row_name.upper():<9}: {list(row)}")
    print("=" * 70)

    # Generate and Save Confusion Matrix Plot
    cm_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_classes)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title(f"ECIE Sentiment Analysis — Confusion Matrix\n({len(target_classes)}-Class Stratified Amazon Reviews)")
    plt.tight_layout()
    plt.savefig(cm_output_path, dpi=300)
    plt.close(fig)
    print(f"\n[✓] Saved Confusion Matrix image to: {cm_output_path}")

    # Save Unified Pipeline Artifact
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"[✓] Saved unified pipeline to: {model_output_path}")

    # Also save separate legacy artifacts for backwards compatibility with predict.py
    legacy_model_path = MODELS_DIR / "sentiment_model.joblib"
    legacy_vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(pipeline.named_steps["clf"], legacy_model_path)
    joblib.dump(pipeline.named_steps["tfidf"], legacy_vec_path)
    print(f"[✓] Saved legacy components for backward compatibility to {MODELS_DIR}")

    # Log to MLflow
    print(f"\n[✓] Logging experiment run to MLflow at: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"TFIDF_SGD_{len(target_classes)}Class_500k"):
        # Log Hyperparameters
        mlflow.log_params({
            "model_type": "SGDClassifier(loss=modified_huber, class_weight=balanced)",
            "vectorizer": "TfidfVectorizer(1-2 grams, sublinear_tf=True)",
            "max_features": max_features,
            "ngram_range": "(1, 2)",
            "sublinear_tf": True,
            "class_weight": "balanced",
            "sample_per_class": sample_per_class,
            "total_samples": len(df),
            "random_state": random_state,
            "test_size": test_size,
            "dataset_classes": ", ".join(target_classes),
        })

        # Log Metrics
        metrics = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "accuracy": test_acc,
            "train_test_gap": acc_gap,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
        for cls_name, p, r, f in zip(target_classes, precisions, recalls, f1s):
            metrics[f"precision_{cls_name}"] = float(p)
            metrics[f"recall_{cls_name}"] = float(r)
            metrics[f"f1_{cls_name}"] = float(f)
        mlflow.log_metrics(metrics)

        # Log Artifacts
        if cm_output_path.exists():
            mlflow.log_artifact(str(cm_output_path))
        if model_output_path.exists():
            mlflow.log_artifact(str(model_output_path))

        print(f"      MLflow run logged successfully!")

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_test_gap": acc_gap,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ECIE 3-Class Sentiment Model Training Pipeline")
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=100000,
        help="Number of reviews to sample per class from PostgreSQL (default: 100000)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Train/test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=200000,
        help="Max TF-IDF feature vocabulary size (default: 200000)",
    )
    args = parser.parse_args()

    train_and_evaluate(
        sample_per_class=args.sample_per_class,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
