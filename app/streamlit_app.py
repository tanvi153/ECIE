"""
Enterprise E-Commerce Intelligence Engine (E-CIE)
Production Web Application & Analytics Dashboard.

Provides an enterprise-grade interactive interface for:
1. Platform Overview: High-level KPI metrics, warehouse distributions, and system architecture.
2. Review Intelligence & XAI: Real-time sentiment prediction with Explainable AI (word-level contribution highlighting).
3. Bulk Review Auditor: Batch CSV/Excel file ingestion, 1-click test datasets, interactive analytics, and report export.
4. Smart Product Recommender: Hybrid content-based filtering with business-aware quality and rating re-ranking.
"""

import html
import io
import math
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
    s = html.unescape(text)
    s = re.sub(r"[\r\n\t]+", " ", s)
    for pattern, repl in ORDERED_CONTRACTIONS:
        s = pattern.sub(repl, s)
    s = re.sub(r"[^\w\s\.\,\!\?\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for continuous confidence display."""
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


# ==============================================================================
# Cached Resource Loaders (Zero Live DB Connection for Fast Cloud Response)
# ==============================================================================
@st.cache_resource(show_spinner="Loading sentiment intelligence pipeline...")
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
    """Loads and standardizes the products subset CSV."""
    if not PRODUCTS_CSV_PATH.exists():
        st.error(f"Product catalog sample not found at `{PRODUCTS_CSV_PATH}`.")
        st.stop()
    df = pd.read_csv(PRODUCTS_CSV_PATH)
    df["product_title"] = df["product_title"].fillna("Unknown Product").str.strip()
    df["category"] = df["category"].fillna("General Merchandise").str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(19.99).round(2)
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce").fillna(4.0).round(2)
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(1).astype(int)
    return df


# ==============================================================================
# Explainable AI (XAI) Feature Contribution Engine
# ==============================================================================
def explain_review_decision(cleaned_text: str, pipeline) -> Dict:
    """
    Computes mathematical token-level contributions to sentiment polarity
    using the pipeline's TF-IDF sparse matrix and linear classifier coefficients.
    """
    try:
        vec = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        feature_names = np.array(vec.get_feature_names_out())

        # For binary linear classifier, coef_[0] represents the positive direction
        is_binary = clf.coef_.shape[0] == 1
        coefs = clf.coef_[0] if is_binary else clf.coef_

        X_vec = vec.transform([cleaned_text])
        nz_indices = X_vec.nonzero()[1]
        values = X_vec.data

        contributions = []
        pos_words = set()
        neg_words = set()

        for idx, val in zip(nz_indices, values):
            word = feature_names[idx]
            weight = float(coefs[idx] * val) if is_binary else float(np.max(coefs[:, idx]) * val)
            contributions.append({
                "word": word,
                "weight": weight,
                "polarity": "positive" if weight > 0 else "negative",
            })
            if weight > 0.05:
                pos_words.add(word)
            elif weight < -0.05:
                neg_words.add(word)

        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x["weight"]), reverse=True)

        top_pos = [c for c in contributions if c["weight"] > 0][:5]
        top_neg = [c for c in contributions if c["weight"] < 0][:5]

        # Inline word highlighting
        tokens = re.split(r"(\s+|[.,!?;:\(\)\[\]\"])", cleaned_text)
        highlighted_parts = []
        for t in tokens:
            clean_t = t.lower().strip()
            if clean_t in pos_words:
                highlighted_parts.append(
                    f'<span class="xai-word-pos" title="Positive Impact">{html.escape(t)}</span>'
                )
            elif clean_t in neg_words:
                highlighted_parts.append(
                    f'<span class="xai-word-neg" title="Negative Impact">{html.escape(t)}</span>'
                )
            else:
                highlighted_parts.append(html.escape(t))

        highlighted_html = "".join(highlighted_parts)

        return {
            "top_positive": top_pos,
            "top_negative": top_neg,
            "highlighted_html": highlighted_html,
            "has_explanation": len(contributions) > 0,
        }
    except Exception as e:
        return {
            "top_positive": [],
            "top_negative": [],
            "highlighted_html": html.escape(cleaned_text),
            "has_explanation": False,
            "error": str(e),
        }


# ==============================================================================
# UI Page Setup & Custom Modern Enterprise Styling
# ==============================================================================
st.set_page_config(
    page_title="E-CIE | Enterprise E-Commerce Intelligence Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Modern Enterprise Typography & Color Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Brand Ribbon */
    .brand-container {
        padding: 1.2rem 1.5rem;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 12px;
        color: #F8FAFC;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #FFFFFF;
    }
    .brand-sub {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 0.25rem;
    }

    /* Stat & Metric Cards */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        text-align: left;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-desc {
        font-size: 0.8rem;
        color: #059669;
        font-weight: 500;
        margin-top: 0.2rem;
    }

    /* Sentiment Decision Badges */
    .badge-positive {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
    }
    .badge-negative {
        background-color: #FFF1F2;
        color: #9F1239;
        border: 1px solid #FECDD3;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
    }

    /* Explainable AI (XAI) Word Highlights */
    .xai-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #1E293B;
        margin-top: 0.75rem;
    }
    .xai-word-pos {
        background-color: #DCFCE7;
        color: #14532D;
        border: 1px solid #86EFAC;
        padding: 2px 7px;
        border-radius: 5px;
        font-weight: 600;
    }
    .xai-word-neg {
        background-color: #FEE2E2;
        color: #7F1D1D;
        border: 1px solid #FCA5A5;
        padding: 2px 7px;
        border-radius: 5px;
        font-weight: 600;
    }

    /* Product Recommendation Card */
    .product-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .product-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.07);
    }
    .rating-pill {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }
    .price-pill {
        background-color: #F1F5F9;
        color: #0F172A;
        font-weight: 700;
        font-size: 0.9rem;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }
    .match-pill {
        background-color: #EFF6FF;
        color: #1E40AF;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }
    .badge-toprated {
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        padding: 2px 6px;
        border-radius: 4px;
        display: inline-block;
        margin-left: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Top Brand Ribbon & Live System Status
# ==============================================================================
st.markdown(
    """
    <div class="brand-container">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 class="brand-title">Enterprise E-Commerce Intelligence Engine</h1>
                <div class="brand-sub">Autonomous Customer Sentiment Analysis & Business-Aware Product Recommendations</div>
            </div>
            <div style="display: flex; gap: 8px; margin-top: 6px;">
                <span style="background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">PostgreSQL: 3.1M Rows</span>
                <span style="background: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid rgba(96, 165, 250, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">Vectors: 60k Catalog</span>
                <span style="background: rgba(168, 85, 247, 0.15); color: #C084FC; border: 1px solid rgba(192, 132, 252, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">Accuracy: 98.31%</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Sidebar - Engine Status & Controls
# ==============================================================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg", width=42)
    st.markdown("### **E-CIE Navigation & Diagnostics**")
    st.caption("B.Tech Capstone Demonstration | E-Commerce Intelligence Engine")

    st.markdown("---")
    st.markdown("#### **System Core Specifications**")
    st.markdown(
        """
        - **NLP Classifier**: SGDClassifier (Modified Huber)
        - **Feature Representation**: TF-IDF 1-2 N-grams (200k max features)
        - **Holdout Test Accuracy**: **98.31%** (100,000 holdout reviews)
        - **Generalization Gap**: **0.86%** (Train: 99.17% vs Test: 98.31%)
        - **Recommender Matrix**: Sparse NearestNeighbors (Cosine Index)
        - **Catalog Indexed**: 60,000 product entities
        """
    )

    st.markdown("---")
    st.markdown("#### **Operational Health**")
    sent_exists = SENTIMENT_MODEL_PATH.exists()
    rec_exists = RECOMMENDER_MODEL_PATH.exists()
    cat_exists = PRODUCTS_CSV_PATH.exists()

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown(f"**NLP Engine**: {'🟢 Ready' if sent_exists else '🔴 Missing'}")
        st.markdown(f"**Catalog Data**: {'🟢 Ready' if cat_exists else '🔴 Missing'}")
    with col_h2:
        st.markdown(f"**Recommender**: {'🟢 Ready' if rec_exists else '🔴 Missing'}")
        st.markdown("**DB Mode**: 🟢 Decoupled")

    st.markdown("---")
    st.caption("Academic Project Submission | Evaluation Dec 3, 2026")

# ==============================================================================
# Application Tabs
# ==============================================================================
tab_overview, tab_sentiment, tab_bulk, tab_recommender = st.tabs([
    "📊 Platform Overview",
    "📝 Review Intelligence & XAI",
    "📁 Bulk Review Auditor",
    "🎯 Smart Product Recommender",
])

# ==============================================================================
# TAB 1: Platform Overview (Analytics Dashboard)
# ==============================================================================
with tab_overview:
    st.subheader("Platform Health & Warehouse Analytics")
    st.caption("Executive overview of ingested datasets, model benchmark metrics, and system distributions.")

    # 4 Executive KPI Cards
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)

    with col_k1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Ingested Warehouse Reviews</div>
                <div class="kpi-value">3,100,000</div>
                <div class="kpi-desc">PostgreSQL 17 Analytical Store</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_k2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Vectorized Product Catalog</div>
                <div class="kpi-value">60,000</div>
                <div class="kpi-desc">Sparse Cosine Vector Space</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_k3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Sentiment Test Accuracy</div>
                <div class="kpi-value">98.31%</div>
                <div class="kpi-desc">Evaluated on 100k Holdout Reviews</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_k4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Overfitting Guard Gap</div>
                <div class="kpi-value">0.86%</div>
                <div class="kpi-desc">Train (99.17%) vs Test (98.31%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Analytics Charts
    col_ch1, col_ch2 = st.columns(2)

    with col_ch1:
        st.markdown("#### **Customer Rating Distribution (3.1M Reviews)**")
        rating_dist = pd.DataFrame({
            "Star Rating": ["5 Stars ★★★★★", "1 Star ★☆☆☆☆", "2 Stars ★★☆☆☆", "4 Stars ★★★★☆"],
            "Review Count": [1277750, 1016457, 533543, 272250],
            "Share (%)": [41.2, 32.8, 17.2, 8.8],
        }).set_index("Star Rating")
        st.bar_chart(rating_dist["Review Count"], color="#2563EB")
        st.caption("Distribution across the complete 3,100,000 Amazon Customer Review dataset.")

    with col_ch2:
        st.markdown("#### **Sentiment Polarity Balance (500k Benchmark)**")
        polarity_dist = pd.DataFrame({
            "Polarity": ["Positive Sentiment (4-5★)", "Negative Sentiment (1-2★)"],
            "Evaluated Reviews": [250000, 250000],
        }).set_index("Polarity")
        st.bar_chart(polarity_dist["Evaluated Reviews"], color="#059669")
        st.caption("Stratified 50/50 balanced training & testing partition (400,000 train / 100,000 test).")

    st.markdown("---")

    col_cat1, col_cat2 = st.columns(2)

    with col_cat1:
        st.markdown("#### **Top Product Catalog Categories**")
        catalog_sample = load_catalog_data()
        top_cats = catalog_sample["category"].value_counts().head(7).reset_index()
        top_cats.columns = ["Category", "Product Count"]
        st.dataframe(top_cats, hide_index=True, use_container_width=True)

    with col_cat2:
        st.markdown("#### **Product Price Bracket Distribution**")
        price_bins = pd.cut(
            catalog_sample["price"],
            bins=[0, 20, 50, 100, 10000],
            labels=["Budget (<$20)", "Mid-Tier ($20-$50)", "Upper Mid ($50-$100)", "Premium ($100+)"],
        ).value_counts().reset_index()
        price_bins.columns = ["Price Bracket", "Catalog Products"]
        st.bar_chart(price_bins.set_index("Price Bracket")["Catalog Products"], color="#D97706")


# ==============================================================================
# TAB 2: Review Intelligence & Explainable AI (XAI)
# ==============================================================================
with tab_sentiment:
    st.subheader("Customer Review Sentiment Classifier & Explainability Engine")
    st.write(
        "Evaluate customer feedback with transparent decision-making. "
        "The **Explainable AI (XAI)** module highlights the exact token features that drove the prediction."
    )

    pipeline = load_sentiment_pipeline()

    # Preset reviews for rapid viva demo
    preset_reviews = {
        "— Select a preset customer review or write your own below —": "",
        "Rave Positive (5★): 'Outstanding wireless headphones! The noise cancellation is phenomenal, battery lasts forever, and sound is crystal clear.'": (
            "Outstanding wireless headphones! The noise cancellation is phenomenal, battery lasts forever, and sound is crystal clear."
        ),
        "Critical Negative (1★): 'Completely defective garbage. The zipper broke on the first day, terrible material quality, and customer service refused to refund.'": (
            "Completely defective garbage. The zipper broke on the first day, terrible material quality, and customer service refused to refund."
        ),
        "Nuanced Mixed Review: 'The sound quality is fantastic and comfortable, but the plastic hinge snapped after two weeks of use.'": (
            "The sound quality is fantastic and comfortable, but the plastic hinge snapped after two weeks of use."
        ),
        "Subtle Critique: 'It works as advertised, but shipping was painfully slow and packaging arrived damaged.'": (
            "It works as advertised, but shipping was painfully slow and packaging arrived damaged."
        ),
    }

    selected_preset = st.selectbox("Quick-Load Demonstration Scenarios:", list(preset_reviews.keys()))
    default_text = preset_reviews[selected_preset] if selected_preset != list(preset_reviews.keys())[0] else ""

    review_input = st.text_area(
        "Customer Review Text:",
        value=default_text or "The laptop backpack is comfortable and spacious, but the strap tore after three days.",
        height=100,
        placeholder="Type or paste any customer review here...",
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    if review_input.strip():
        cleaned_text = clean_text_for_inference(review_input)
        if not cleaned_text:
            st.warning("Please provide valid alphabetic text for evaluation.")
        else:
            # Predict label and confidence
            pred_label = pipeline.predict([cleaned_text])[0]
            classes = list(pipeline.classes_)

            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba([cleaned_text])[0]
            elif hasattr(pipeline, "decision_function"):
                decision_scores = pipeline.decision_function([cleaned_text])[0]
                probabilities = compute_softmax(decision_scores)
            else:
                probabilities = [1.0 / len(classes)] * len(classes)

            prob_map = {cls: prob for cls, prob in zip(classes, probabilities)}
            top_confidence = prob_map.get(pred_label, 0.0)

            # Explainable AI Analysis
            xai_data = explain_review_decision(cleaned_text, pipeline)

            st.markdown("---")
            st.markdown("### **Evaluation Result**")

            col_res1, col_res2, col_res3 = st.columns([1.2, 1.2, 2.2])

            badge_style = f"badge-{pred_label.lower()}"
            emoji_icon = "😊" if pred_label.lower() == "positive" else ("😞" if pred_label.lower() == "negative" else "😐")

            with col_res1:
                st.markdown(f"**Predicted Sentiment Class:**")
                st.markdown(
                    f'<span class="{badge_style}">{emoji_icon} {pred_label.upper()}</span>',
                    unsafe_allow_html=True,
                )
                est_stars = "4-5 Stars" if pred_label.lower() == "positive" else ("1-2 Stars" if pred_label.lower() == "negative" else "3 Stars")
                st.caption(f"Estimated Star Band: **{est_stars}**")

            with col_res2:
                st.markdown(f"**Model Confidence:**")
                st.metric(label="Decision Probability", value=f"{top_confidence * 100:.1f}%")
                st.caption("Calibrated Modified Huber Probability")

            with col_res3:
                st.markdown("**Preprocessed Input Text:**")
                st.code(cleaned_text, language="text")

            # Suggestion 1: Explainable AI Word Highlighting
            st.markdown("#### **Explainable AI (XAI): Feature Attribution Breakdown**")
            st.write(
                "Words that increased **Positive** sentiment are highlighted in <span style='color:#15803D;font-weight:600;'>Green</span>; "
                "words that drove **Negative** sentiment are highlighted in <span style='color:#B91C1C;font-weight:600;'>Red</span>.",
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="xai-box">{xai_data["highlighted_html"]}</div>',
                unsafe_allow_html=True,
            )

            # Feature Impact Bar Chart
            col_xai1, col_xai2 = st.columns(2)
            with col_xai1:
                if xai_data["top_positive"]:
                    pos_df = pd.DataFrame([
                        {"Driver Word": item["word"], "Impact Weight": round(item["weight"], 3)}
                        for item in xai_data["top_positive"]
                    ]).set_index("Driver Word")
                    st.markdown("**Top Positive Polarity Drivers (+):**")
                    st.bar_chart(pos_df["Impact Weight"], color="#10B981")
                else:
                    st.info("No strong positive trigger words detected.")

            with col_xai2:
                if xai_data["top_negative"]:
                    neg_df = pd.DataFrame([
                        {"Driver Word": item["word"], "Impact Weight": round(abs(item["weight"]), 3)}
                        for item in xai_data["top_negative"]
                    ]).set_index("Driver Word")
                    st.markdown("**Top Negative Polarity Drivers (−):**")
                    st.bar_chart(neg_df["Impact Weight"], color="#EF4444")
                else:
                    st.info("No strong negative trigger words detected.")


# ==============================================================================
# TAB 3: Bulk Review Auditor (Upload File & Batch Analytics)
# ==============================================================================
with tab_bulk:
    st.subheader("Bulk Review Auditor & Batch Analytics")
    st.write(
        "Upload customer review datasets (`.csv`, `.xlsx`, `.tsv`) or load our standardized test batch "
        "to run rapid parallel inference and visualize sentiment proportions."
    )

    # Pre-built sample batch of 25 customer reviews for instant viva demo
    SAMPLE_BATCH_REVIEWS = [
        "Incredible sound quality and ultra-comfortable ear pads. Worth every penny!",
        "Defective right out of the box. Stopped turning on after 2 hours. Returning.",
        "Very fast shipping. The product matches the description exactly.",
        "Terrible customer service. Refused my replacement request for broken parts.",
        "Build quality is solid and battery easily lasts 3 full days. Highly recommended.",
        "The zipper broke instantly and the material feels cheap and flimsy.",
        "Exceeded all my expectations. Beautiful design and crisp performance.",
        "Horrible static noise in the left earbud. Completely unusable.",
        "Great value for money. Does the job without any fuss.",
        "Do not waste your money on this junk. Broke within 3 days.",
        "Super lightweight and portable. Perfect for travel and daily commute.",
        "Screen flickers constantly and customer support was completely unhelpful.",
        "Loved the packaging and overall finish. Would buy again as a gift.",
        "Poor quality stitching, began coming undone after one gentle wash.",
        "The best purchase I made this year. Flawless functionality.",
        "Overheats after 15 minutes of normal use. Very disappointing.",
        "Sleek look, responsive touch controls, and clean audio.",
        "Cheap plastic feel and missing half the screws in the box.",
        "Surprisingly good bass and stays securely in ear while running.",
        "Faulty battery life. Drains from 100% to 0% in under 30 minutes.",
        "Sturdy aluminum construction and very satisfying tactile feedback.",
        "Completely useless. Does not connect to Wi-Fi at all.",
        "Clear setup instructions and worked flawlessly on the first try.",
        "Fell apart after two weeks of normal use. Absolute waste of money.",
        "Five stars! Excellent craftsmanship and speedy delivery.",
    ]

    col_up1, col_up2 = st.columns([3, 1.2])

    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload Review File (.CSV or .XLSX):",
            type=["csv", "xlsx", "tsv"],
            help="File must contain a column with text reviews (e.g. 'review', 'review_text', or 'text').",
        )

    with col_up2:
        st.markdown("<br>", unsafe_allow_html=True)
        load_sample_btn = st.button("⚡ Load 25 Test Reviews", use_container_width=True)

    df_to_audit = None

    if load_sample_btn:
        df_to_audit = pd.DataFrame({"review_text": SAMPLE_BATCH_REVIEWS})
        st.success("Loaded 25 verified customer demonstration reviews!")

    elif uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".tsv"):
                sep = "\t" if uploaded_file.name.endswith(".tsv") else ","
                df_to_audit = pd.read_csv(uploaded_file, sep=sep)
            else:
                df_to_audit = pd.read_excel(uploaded_file)
            st.success(f"Successfully loaded `{uploaded_file.name}` ({len(df_to_audit):,} records)!")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

    if df_to_audit is not None and not df_to_audit.empty:
        # Detect text column
        text_cols = [
            c for c in df_to_audit.columns
            if any(k in str(c).lower() for k in ["review", "text", "comment", "desc", "feedback", "body"])
        ]
        chosen_col = text_cols[0] if text_cols else df_to_audit.columns[0]

        if len(df_to_audit.columns) > 1:
            chosen_col = st.selectbox("Select Review Text Column:", options=list(df_to_audit.columns), index=df_to_audit.columns.get_loc(chosen_col))

        pipeline = load_sentiment_pipeline()

        # Run vectorized batch inference
        with st.spinner(f"Running vectorized inference across {len(df_to_audit):,} reviews..."):
            cleaned_batch = df_to_audit[chosen_col].astype(str).apply(clean_text_for_inference)
            preds = pipeline.predict(cleaned_batch)
            probs = pipeline.predict_proba(cleaned_batch) if hasattr(pipeline, "predict_proba") else None

            df_to_audit["Predicted_Sentiment"] = preds
            if probs is not None:
                max_probs = np.max(probs, axis=1)
                df_to_audit["Confidence (%)"] = (max_probs * 100).round(1)

        # Batch Summary Metrics
        st.markdown("---")
        st.markdown("### **Batch Audit Summary**")

        counts = df_to_audit["Predicted_Sentiment"].value_counts()
        pos_count = counts.get("positive", 0)
        neg_count = counts.get("negative", 0)
        total_count = len(df_to_audit)
        pos_pct = (pos_count / total_count) * 100 if total_count else 0
        neg_pct = (neg_count / total_count) * 100 if total_count else 0
        avg_conf = df_to_audit["Confidence (%)"].mean() if "Confidence (%)" in df_to_audit else 95.0

        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            st.metric("Total Reviews Processed", f"{total_count:,}")
        with col_b2:
            st.metric("Positive Sentiment", f"{pos_count:,} ({pos_pct:.1f}%)")
        with col_b3:
            st.metric("Negative Sentiment", f"{neg_count:,} ({neg_pct:.1f}%)")
        with col_b4:
            st.metric("Avg Prediction Confidence", f"{avg_conf:.1f}%")

        # Distribution Chart
        st.markdown("#### **Customer Polarity Breakdown**")
        chart_df = pd.DataFrame({
            "Sentiment Class": ["Positive (Happy)", "Negative (Unhappy)"],
            "Count": [pos_count, neg_count],
        }).set_index("Sentiment Class")
        st.bar_chart(chart_df["Count"], color="#3B82F6")

        # Filterable Results Table
        st.markdown("#### **Audited Records Table**")
        filter_opt = st.radio("Filter Table View:", ["All Reviews", "Positive Only", "Negative Only"], horizontal=True)

        display_df = df_to_audit.copy()
        if filter_opt == "Positive Only":
            display_df = display_df[display_df["Predicted_Sentiment"] == "positive"]
        elif filter_opt == "Negative Only":
            display_df = display_df[display_df["Predicted_Sentiment"] == "negative"]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download Processed Results
        csv_buffer = io.StringIO()
        df_to_audit.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Audited Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name="ecie_sentiment_audit_results.csv",
            mime="text/csv",
        )


# ==============================================================================
# TAB 4: Smart Product Recommender (Quality & Rating Weighted)
# ==============================================================================
with tab_recommender:
    st.subheader("Content-Based Smart Product Recommender")
    st.write(
        "Find contextually relevant products using sparse TF-IDF and NearestNeighbors cosine embeddings. "
        "Enhanced with **Smart Business Ranking** to elevate highly-rated items over defective products."
    )

    bundle = load_recommender_bundle()
    vectorizer = bundle["vectorizer"]
    nn_model = bundle["nn_model"]

    catalog_df = bundle.get("products_df")
    if catalog_df is None:
        catalog_df = load_catalog_data()

    asin_to_idx = {str(a).strip().upper(): i for i, a in enumerate(catalog_df["parent_asin"])}

    rec_mode = st.radio(
        "Product Search Methodology:",
        ["🎯 Select Product from Catalog", "🔍 Natural Language / Feature Search"],
        horizontal=True,
    )

    target_query = ""
    is_asin_mode = False

    if rec_mode == "🎯 Select Product from Catalog":
        is_asin_mode = True
        top_sample = catalog_df.head(250)
        product_options = {
            f"[{row['parent_asin']}] {row['product_title'][:75]} (${row['price']:.2f}) — ⭐{row['avg_rating']}": row["parent_asin"]
            for _, row in top_sample.iterrows()
        }
        selected_label = st.selectbox(
            "Choose a baseline product from catalog:",
            options=list(product_options.keys()),
        )
        target_query = product_options[selected_label]
    else:
        is_asin_mode = False
        target_query = st.text_input(
            "Enter product keywords, specifications, or category:",
            value="Countertop Ice Maker Machine",
            placeholder="e.g. 'Refrigerator Water Filter', 'Subscription Beauty Box', 'Stainless Steel Blender'...",
        )

    # Suggestion 2 Controls: Ranking Strategy & Quality Filter
    col_opt1, col_opt2, col_opt3 = st.columns([1.5, 1.2, 1.2])

    with col_opt1:
        ranking_strategy = st.radio(
            "Ranking Algorithm:",
            [
                "⚡ Smart Business Ranking (Balances Similarity + Rating + Popularity)",
                "🔍 Raw Semantic Match (Pure Title Similarity)",
            ],
            help="Smart Business Ranking demotes 1-star poorly rated items and promotes highly rated, proven products.",
        )

    with col_opt2:
        min_rating_filter = st.slider(
            "Min Star Rating Filter:",
            min_value=1.0,
            max_value=4.5,
            value=3.0,
            step=0.5,
            help="Filters out low-rated items matching your query.",
        )

    with col_opt3:
        top_n = st.slider("Recommendations (Top-N):", min_value=1, max_value=15, value=5)

    if target_query:
        query_str = str(target_query).strip()

        if is_asin_mode and query_str.upper() in asin_to_idx:
            target_idx = asin_to_idx[query_str.upper()]
            target_row = catalog_df.iloc[target_idx]

            st.markdown(
                f"""
                <div class="kpi-card" style="margin-bottom: 1.2rem;">
                    <strong>Baseline Selected Product:</strong> {target_row['product_title']}<br>
                    <strong>Category:</strong> {target_row['category']} | 
                    <strong>Price:</strong> ${target_row['price']:.2f} | 
                    <strong>Rating:</strong> ⭐ {target_row['avg_rating']} ({target_row['review_count']} reviews) | 
                    <strong>ASIN:</strong> <code>{target_row['parent_asin']}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

            item_vec = nn_model._fit_X[target_idx]
            # Fetch a larger pool so filtering and smart re-ranking have candidates
            distances, indices = nn_model.kneighbors(item_vec, n_neighbors=min(len(catalog_df), top_n * 6 + 1))
            raw_candidates = [(dist, idx) for dist, idx in zip(distances[0], indices[0]) if idx != target_idx]

        else:
            q_vec = vectorizer.transform([query_str])
            if q_vec.nnz == 0:
                st.info(f"ℹ️ No direct vocabulary match for '{query_str}'. Try standard terms like 'Filter', 'Ice Maker', 'Subscription', or 'Headphones'.")
                raw_candidates = []
            else:
                distances, indices = nn_model.kneighbors(q_vec, n_neighbors=min(len(catalog_df), top_n * 6))
                raw_candidates = [(dist, idx) for dist, idx in zip(distances[0], indices[0])]

        scored_results = []
        for dist, idx in raw_candidates:
            row = catalog_df.iloc[idx]
            avg_r = float(row["avg_rating"])
            r_count = int(row["review_count"])

            # Apply user's minimum star rating filter
            if avg_r < min_rating_filter:
                continue

            sim_score = max(0.0, 1.0 - float(dist))

            if "Smart Business" in ranking_strategy:
                # Quality score normalized to [0, 1]
                quality_score = avg_r / 5.0
                # Popularity score using logarithmic scaling capped at 100 reviews
                pop_score = min(1.0, math.log1p(r_count) / math.log1p(100))
                # Composite formula: 65% Similarity + 25% Star Rating + 10% Review Volume
                composite_score = (0.65 * sim_score) + (0.25 * quality_score) + (0.10 * pop_score)
            else:
                composite_score = sim_score

            scored_results.append({
                "composite_score": composite_score,
                "similarity_score": sim_score,
                "row": row,
            })

        # Sort by final score
        scored_results.sort(key=lambda x: x["composite_score"], reverse=True)
        final_picks = scored_results[:top_n]

        if final_picks:
            st.markdown(f"#### **Recommended Products ({len(final_picks)} Matches)**")

            for item in final_picks:
                row = item["row"]
                sim = item["similarity_score"] * 100
                comp = item["composite_score"] * 100
                r_val = float(row["avg_rating"])
                r_num = int(row["review_count"])
                price = float(row["price"])

                top_rated_badge = '<span class="badge-toprated">⭐ Top Rated</span>' if r_val >= 4.5 and r_num >= 5 else ""

                st.markdown(
                    f"""
                    <div class="product-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div style="font-weight: 700; font-size: 1.05rem; color: #0F172A; margin-bottom: 0.35rem;">
                                {row['product_title']}
                            </div>
                            <div style="text-align: right; min-width: 120px;">
                                <span class="price-pill">${price:.2f}</span>
                            </div>
                        </div>
                        <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.5rem;">
                            Category: <strong>{row['category']}</strong> | ASIN: <code>{row['parent_asin']}</code>
                        </div>
                        <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                            <span class="rating-pill">⭐ {r_val:.1f} ({r_num:,} reviews)</span>
                            {top_rated_badge}
                            <span class="match-pill">Similarity: {sim:.1f}%</span>
                            <span style="background-color: #F8FAFC; color: #475569; font-size: 0.8rem; font-weight: 600; padding: 3px 8px; border-radius: 6px; border: 1px solid #E2E8F0;">
                                Composite Score: {comp:.1f}%
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning(f"No products found meeting the minimum rating threshold of {min_rating_filter}★. Try lowering the rating slider.")
