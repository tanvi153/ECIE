"""
Enterprise E-Commerce Intelligence Engine (E-CIE)
Interactive Web Application.

A lightweight, zero-database Streamlit dashboard for:
1. Multi-Class Sentiment Analysis (Negative, Neutral, Positive)
2. Content-Based Product Recommendation Engine (Cosine Similarity over TF-IDF)

Built for fast deployment on Streamlit Community Cloud and Docker containers.
"""

import html
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# Path Setup & Configuration
# ==============================================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SENTIMENT_MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
RECOMMENDER_MODEL_PATH = PROJECT_ROOT / "models" / "recommender.joblib"
PRODUCTS_CSV_PATH = PROJECT_ROOT / "data" / "products_sample.csv"

# ==============================================================================
# Contraction Rules for Inference Preprocessing
# ==============================================================================
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


def clean_text_for_inference(text: str) -> str:
    """Sanitizes review text matching the training pipeline preprocessing."""
    if not isinstance(text, str):
        return ""
    # 1. Unescape HTML entities
    s = html.unescape(text)
    # 2. Normalize whitespace
    s = re.sub(r"[\r\n\t]+", " ", s)
    # 3. Expand contractions in strict order
    for pattern, repl in ORDERED_CONTRACTIONS:
        s = pattern.sub(repl, s)
    # 4. Remove special characters and excess whitespace
    s = re.sub(r"[^\w\s\.\,\!\?\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for continuous confidence display."""
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


# ==============================================================================
# Cached Resource Loaders (Zero DB Connection)
# ==============================================================================
@st.cache_resource(show_spinner="Loading sentiment model pipeline...")
def load_sentiment_pipeline():
    """Loads the trained scikit-learn sentiment pipeline."""
    if not SENTIMENT_MODEL_PATH.exists():
        st.error(f"Sentiment pipeline not found at `{SENTIMENT_MODEL_PATH}`. Please run training first.")
        st.stop()
    return joblib.load(SENTIMENT_MODEL_PATH)


@st.cache_resource(show_spinner="Loading product recommendation engine...")
def load_recommender_bundle():
    """Loads the TF-IDF vectorizer and NearestNeighbors cosine index."""
    if not RECOMMENDER_MODEL_PATH.exists():
        st.error(f"Recommender bundle not found at `{RECOMMENDER_MODEL_PATH}`. Please run training first.")
        st.stop()
    return joblib.load(RECOMMENDER_MODEL_PATH)


@st.cache_data(show_spinner="Loading product catalog sample...")
def load_catalog_data():
    """Loads the products subset CSV."""
    if not PRODUCTS_CSV_PATH.exists():
        st.error(f"Product catalog sample not found at `{PRODUCTS_CSV_PATH}`.")
        st.stop()
    df = pd.read_csv(PRODUCTS_CSV_PATH)
    df["product_title"] = df["product_title"].fillna("Unknown Product").str.strip()
    df["category"] = df["category"].fillna("General").str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(19.99).round(2)
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce").fillna(4.0).round(2)
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(1).astype(int)
    return df


# ==============================================================================
# UI Page Setup
# ==============================================================================
st.set_page_config(
    page_title="E-CIE | Enterprise E-Commerce Intelligence Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .badge-positive {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-neutral {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-negative {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Sidebar - Project Overview & Health
# ==============================================================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg", width=50)
    st.title("E-CIE Engine")
    st.caption("**Enterprise E-Commerce Intelligence Engine**")
    st.caption("B.Tech Minor Project Live Demonstration")

    st.markdown("---")
    st.markdown("### 📌 Architecture Summary")
    st.markdown(
        """
        - **Sentiment Analysis**:
          - 3-Class Linear Classifier (`SGDClassifier` with TF-IDF)
          - Labels strictly derived from star ratings (1-2: Neg, 3: Neu, 4-5: Pos)
          - Evaluated test accuracy: **74.84%** (honest baseline without keyword leakage)
        - **Recommendation Engine**:
          - Content-based TF-IDF (1-2 grams) on Title + Category
          - Brute-force Cosine Similarity via sparse `NearestNeighbors`
          - **60,000 product** catalog subset (6.56 MB)
        - **Deployment**:
          - Decoupled from live PostgreSQL for cloud memory-safety
        """
    )

    st.markdown("---")
    st.markdown("### ⚙️ Engine Status")

    # Check artifacts existence
    sent_ready = SENTIMENT_MODEL_PATH.exists()
    rec_ready = RECOMMENDER_MODEL_PATH.exists()
    cat_ready = PRODUCTS_CSV_PATH.exists()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"**Sentiment**: {'🟢 Ready' if sent_ready else '🔴 Missing'}")
        st.markdown(f"**Catalog**: {'🟢 Ready' if cat_ready else '🔴 Missing'}")
    with col_s2:
        st.markdown(f"**Recommender**: {'🟢 Ready' if rec_ready else '🔴 Missing'}")
        st.markdown("**DB Mode**: 🟡 Decoupled")

    st.markdown("---")
    st.caption("Developed by Tanvi Singla | E-CIE Team")

# ==============================================================================
# Main Application Content (Tabs)
# ==============================================================================
st.markdown('<div class="main-header">Enterprise E-Commerce Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Real-time NLP sentiment evaluation and content-based recommendation system</div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["📝 Sentiment Tester", "🎯 Product Recommender"])

# ------------------------------------------------------------------------------
# Tab 1: Sentiment Tester
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Customer Review Sentiment Classifier")
    st.write(
        "Enter any raw customer review or select a preset example to evaluate predicted sentiment "
        "and class confidence distribution."
    )

    # Preset examples for rapid demonstration
    example_options = {
        "— Select a preset review or write your own —": "",
        "Positive (5★): 'Absolutely fantastic! Exceeded all expectations, great build quality and fast shipping.'": (
            "Absolutely fantastic! Exceeded all expectations, great build quality and fast shipping."
        ),
        "Neutral (3★): 'It is okay. Does what it says, but nothing particularly impressive for the price.'": (
            "It is okay. Does what it says, but nothing particularly impressive for the price."
        ),
        "Negative (1★): 'Total waste of money. Broke on the second day and customer support was completely unhelpful.'": (
            "Total waste of money. Broke on the second day and customer support was completely unhelpful."
        ),
        "Subtle Negation: 'I did not hate it, but I will definitely never buy this brand again.'": (
            "I did not hate it, but I will definitely never buy this brand again."
        ),
    }

    selected_example = st.selectbox("Quick-Load Example Review:", list(example_options.keys()))
    preset_text = example_options[selected_example]

    user_input = st.text_area(
        "Enter Review Text:",
        value=preset_text,
        height=120,
        placeholder="Type or paste a product review here (e.g., 'The motor makes a strange loud rattling sound...')...",
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_clicked or (user_input and preset_text):
        cleaned_input = clean_text_for_inference(user_input)

        if len(cleaned_input) < 3:
            st.warning("⚠️ Review input is too short or empty. Please enter at least 3 characters.")
        else:
            pipeline = load_sentiment_pipeline()

            # Predict label and decision score
            pred_label = pipeline.predict([cleaned_input])[0]
            classes = list(pipeline.classes_)

            if hasattr(pipeline, "decision_function"):
                decision_scores = pipeline.decision_function([cleaned_input])[0]
                probabilities = compute_softmax(decision_scores)
            elif hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba([cleaned_input])[0]
            else:
                probabilities = [1.0 / len(classes)] * len(classes)

            prob_map = {cls: prob for cls, prob in zip(classes, probabilities)}

            # Color styling by class
            badge_class = f"badge-{pred_label.lower()}"
            emoji_map = {"positive": "😊", "neutral": "😐", "negative": "😞"}
            star_map = {"positive": "4-5 Stars", "neutral": "3 Stars", "negative": "1-2 Stars"}

            st.markdown("---")
            col_res1, col_res2, col_res3 = st.columns([1.5, 1.5, 2])

            with col_res1:
                st.markdown(f"### Predicted Class\n")
                st.markdown(
                    f'<span class="{badge_class}">{emoji_map.get(pred_label, "")} {pred_label.upper()}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Estimated Star Band: **{star_map.get(pred_label, 'N/A')}**")

            with col_res2:
                top_confidence = prob_map.get(pred_label, 0.0)
                st.metric("Model Confidence", f"{top_confidence * 100:.1f}%")

            with col_res3:
                st.markdown("**Preprocessed Input Text:**")
                st.caption(f"`{cleaned_input}`")

            # Probability Distribution Chart
            st.markdown("#### Class Confidence Distribution")
            chart_df = pd.DataFrame(
                {
                    "Sentiment Class": [c.capitalize() for c in classes],
                    "Confidence (%)": [round(prob_map[c] * 100, 2) for c in classes],
                }
            ).set_index("Sentiment Class")

            st.bar_chart(chart_df, horizontal=True)

# ------------------------------------------------------------------------------
# Tab 2: Product Recommender
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Content-Based Product Recommender")
    st.write(
        "Find similar products based on catalog metadata (Title, Category, and Hierarchy) "
        "using TF-IDF feature embeddings and Cosine Distance."
    )

    bundle = load_recommender_bundle()
    vectorizer = bundle["vectorizer"]
    nn_model = bundle["nn_model"]

    # Prefer catalog from bundle if embedded, otherwise load CSV
    catalog_df = bundle.get("products_df")
    if catalog_df is None:
        catalog_df = load_catalog_data()

    # Build ASIN lookup dictionary
    asin_to_idx = {str(a).strip().upper(): i for i, a in enumerate(catalog_df["parent_asin"])}

    rec_mode = st.radio(
        "Search Mode:",
        ["🎯 Select Product from Catalog", "🔍 Natural Language / Keyword Search"],
        horizontal=True,
    )

    target_query = ""
    is_asin_mode = False

    if rec_mode == "🎯 Select Product from Catalog":
        is_asin_mode = True
        # Offer popular sample products
        top_sample = catalog_df.head(250)
        product_options = {
            f"[{row['parent_asin']}] {row['product_title'][:80]} (${row['price']:.2f})": row["parent_asin"]
            for _, row in top_sample.iterrows()
        }

        selected_label = st.selectbox(
            "Choose a product to find similar recommendations:",
            options=list(product_options.keys()),
        )
        target_query = product_options[selected_label]

    else:
        is_asin_mode = False
        target_query = st.text_input(
            "Enter product keywords, features, or category:",
            value="Countertop Ice Maker Machine",
            placeholder="e.g. 'Refrigerator Water Filter', 'Subscription Beauty Box', 'Jazz Guitar'...",
        )

    col_rec_opt1, col_rec_opt2 = st.columns([1, 4])
    with col_rec_opt1:
        top_n = st.slider("Number of Recommendations (Top-N):", min_value=1, max_value=15, value=5)

    if target_query:
        query_str = str(target_query).strip()

        if is_asin_mode and query_str.upper() in asin_to_idx:
            # Query by ASIN
            target_idx = asin_to_idx[query_str.upper()]
            target_row = catalog_df.iloc[target_idx]

            st.markdown(
                f"""
                <div class="metric-card">
                    <strong>Selected Product:</strong> {target_row['product_title']}<br>
                    <strong>Category:</strong> {target_row['category']} | <strong>Price:</strong> ${target_row['price']:.2f} | 
                    <strong>Rating:</strong> ⭐ {target_row['avg_rating']} ({target_row['review_count']} reviews)
                </div>
                """,
                unsafe_allow_html=True,
            )

            item_vec = nn_model._fit_X[target_idx]
            distances, indices = nn_model.kneighbors(item_vec, n_neighbors=top_n + 1)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == target_idx:
                    continue
                row = catalog_df.iloc[idx]
                sim_score = round(max(0.0, 1.0 - float(dist)), 4)
                results.append({
                    "Similarity": f"{sim_score * 100:.1f}%",
                    "Similarity Score": sim_score,
                    "Product Title": row["product_title"],
                    "Category": row["category"],
                    "Price ($)": f"${row['price']:.2f}",
                    "Rating": f"⭐ {row['avg_rating']} ({row['review_count']})",
                    "ASIN": row["parent_asin"],
                })
                if len(results) == top_n:
                    break

        else:
            # Free-text search
            q_vec = vectorizer.transform([query_str])
            if q_vec.nnz == 0:
                st.info(f"ℹ️ No direct vocabulary match for '{query_str}'. Try different keywords like 'Filter', 'Ice Maker', or 'Magazine'.")
                results = []
            else:
                distances, indices = nn_model.kneighbors(q_vec, n_neighbors=top_n)
                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    row = catalog_df.iloc[idx]
                    sim_score = round(max(0.0, 1.0 - float(dist)), 4)
                    results.append({
                        "Similarity": f"{sim_score * 100:.1f}%",
                        "Similarity Score": sim_score,
                        "Product Title": row["product_title"],
                        "Category": row["category"],
                        "Price ($)": f"${row['price']:.2f}",
                        "Rating": f"⭐ {row['avg_rating']} ({row['review_count']})",
                        "ASIN": row["parent_asin"],
                    })

        if results:
            st.markdown(f"#### Top {len(results)} Recommendations")
            res_df = pd.DataFrame(results)

            # Display table with clean columns
            st.dataframe(
                res_df[["Similarity", "Product Title", "Category", "Price ($)", "Rating", "ASIN"]],
                use_container_width=True,
                hide_index=True,
            )
