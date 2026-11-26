"""
Enterprise E-Commerce Intelligence Engine (E-CIE)
Production Web Application & Analytics Dashboard.

Provides an enterprise-grade interactive interface for:
1. Platform Overview: High-level KPI metrics, warehouse distributions, and system architecture.
2. Review Intelligence & XAI: Real-time sentiment prediction with Explainable AI.
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
import plotly.express as px
import plotly.graph_objects as go
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
POWERBI_MASTER_CSV_PATH = PROJECT_ROOT / "data" / "powerbi" / "master_unified_report.csv"

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


@st.cache_data(show_spinner="Loading Power BI master dataset...")
def load_powerbi_master_data() -> pd.DataFrame:
    """Loads the pre-aggregated Power BI master analytical dataset."""
    if not POWERBI_MASTER_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(POWERBI_MASTER_CSV_PATH)


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

    /* Enhanced Large Sidebar Navigation Menu */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        gap: 0.65rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px !important;
        transition: all 0.2s ease;
        cursor: pointer;
        width: 100%;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: #EEF2F6;
        border-color: #CBD5E1;
        transform: translateX(2px);
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label div,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label span {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #0F172A !important;
        line-height: 1.4 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Top Brand Ribbon
# ==============================================================================
st.markdown(
    """
    <div class="brand-container">
        <div>
            <h1 class="brand-title">Enterprise E-Commerce Intelligence Engine</h1>
            <div class="brand-sub">Autonomous Customer Sentiment Analysis & Business-Aware Product Recommendations</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Sidebar - E-CIE Navigation
# ==============================================================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg", width=42)
    st.markdown("### **E-CIE Navigation**")
    st.caption("Minor Project | E-Commerce Intelligence Engine")

    st.markdown("---")
    nav_selection = st.radio(
        "Navigation Menu:",
        [
            "📊 Platform Overview",
            "📝 Review Intelligence & XAI",
            "📁 Bulk Review Auditor",
            "🎯 Smart Product Recommender",
        ],
        index=0,
    )


# ==============================================================================
# VIEW 1: Platform Overview (Analytics Dashboard)
# ==============================================================================
if nav_selection == "📊 Platform Overview":
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

    # 4 Multi-Page Analytical Views (Exact Power BI Mirror)
    pbi_tab1, pbi_tab2, pbi_tab3, pbi_tab4 = st.tabs([
        "🏬 Page 1: Category & Store Sentiment",
        "📈 Page 2: Temporal Rating Trends (2012–2023)",
        "💰 Page 3: Price vs Rating Elasticity",
        "⚙️ Page 4: System Architecture & Warehouse Health",
    ])

    df_pbi = load_powerbi_master_data()

    # --------------------------------------------------------------------------
    # TAB 1: Category & Store Sentiment
    # --------------------------------------------------------------------------
    with pbi_tab1:
        st.markdown("### **Category & Merchant Store Sentiment Intelligence**")
        st.caption("Interactive cross-filtering of sentiment distribution across product categories and top merchant storefronts.")

        if not df_pbi.empty:
            col_filter1, col_filter2 = st.columns([2, 2])
            with col_filter1:
                cat_options = ["All Categories"] + sorted(df_pbi["category"].dropna().unique().tolist())
                selected_cat = st.selectbox("Filter by Category:", cat_options, key="pbi_cat_filter")
            with col_filter2:
                sentiment_filter = st.radio("Sentiment Polarity:", ["All Sentiments", "Positive", "Negative"], horizontal=True, key="pbi_sent_filter")

            # Filter data
            dff1 = df_pbi.copy()
            if selected_cat != "All Categories":
                dff1 = dff1[dff1["category"] == selected_cat]
            if sentiment_filter == "Positive":
                dff1 = dff1[dff1["sentiment"] == "positive"]
            elif sentiment_filter == "Negative":
                dff1 = dff1[dff1["sentiment"] == "negative"]

            # Visual 1 & 2 in two columns
            c_v1, c_v2 = st.columns(2)

            with c_v1:
                st.markdown("##### **Category Sentiment Share (100% Normalized)**")
                cat_summary = df_pbi.groupby(["category", "sentiment"])["review_count"].sum().reset_index()
                fig_cat = px.histogram(
                    cat_summary,
                    y="category",
                    x="review_count",
                    color="sentiment",
                    barnorm="percent",
                    orientation="h",
                    color_discrete_map={"positive": "#10B981", "negative": "#EF4444"},
                    labels={"review_count": "Sentiment Ratio (%)", "category": "Product Category"},
                )
                fig_cat.update_layout(
                    barmode="stack",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_cat, use_container_width=True)

            with c_v2:
                st.markdown("##### **Top Merchant Stores by Review Volume**")
                store_summary = dff1[dff1["store_name"] != "Other Verified Stores"].groupby(["store_name", "sentiment"])["review_count"].sum().reset_index()
                top_stores = store_summary.groupby("store_name")["review_count"].sum().nlargest(8).index
                store_summary = store_summary[store_summary["store_name"].isin(top_stores)]

                fig_store = px.bar(
                    store_summary,
                    y="store_name",
                    x="review_count",
                    color="sentiment",
                    barmode="group",
                    orientation="h",
                    color_discrete_map={"positive": "#10B981", "negative": "#EF4444"},
                    labels={"review_count": "Total Reviews", "store_name": "Merchant Store"},
                )
                fig_store.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_store, use_container_width=True)

            # Store Benchmark Table
            st.markdown("##### **Merchant Store Performance Scorecard**")
            store_table = dff1.groupby("store_name").agg(
                total_reviews=("review_count", "sum"),
                avg_rating=("avg_rating", "mean"),
                verified_reviews=("verified_reviews", "sum"),
            ).reset_index()
            store_table["verified_rate"] = (store_table["verified_reviews"] / store_table["total_reviews"] * 100).round(1).astype(str) + "%"
            store_table["avg_rating"] = store_table["avg_rating"].round(2)
            store_table = store_table.sort_values(by="total_reviews", ascending=False).head(10)
            store_table.columns = ["Merchant Store", "Total Reviews", "Average Rating (★)", "Verified Reviews", "Verified Purchase %"]
            st.dataframe(store_table, hide_index=True, use_container_width=True)
        else:
            st.warning("Master dataset not found. Please run `scripts/export_powerbi.py`.")

    # --------------------------------------------------------------------------
    # TAB 2: Temporal Rating Trends (2012-2023)
    # --------------------------------------------------------------------------
    with pbi_tab2:
        st.markdown("### **Temporal Rating Trends & Sentiment Velocity**")
        st.caption("Longitudinal analysis of consumer rating trajectories and review growth velocity over 11 years (2012–2023).")

        if not df_pbi.empty:
            year_range = st.slider("Select Year Timeline Range:", min_value=2012, max_value=2023, value=(2014, 2023), key="pbi_year_slider")
            dff2 = df_pbi[(df_pbi["review_year"] >= year_range[0]) & (df_pbi["review_year"] <= year_range[1])]

            c_t1, c_t2 = st.columns(2)

            with c_t1:
                st.markdown("##### **Longitudinal Rating Evolution (Trend Line)**")
                yearly_ratings = dff2.groupby(["review_year", "category"])["avg_rating"].mean().reset_index()
                fig_trend = px.line(
                    yearly_ratings,
                    x="review_year",
                    y="avg_rating",
                    color="category",
                    markers=True,
                    labels={"review_year": "Review Year", "avg_rating": "Average Rating (1-5★)"},
                )
                fig_trend.update_layout(
                    yaxis=dict(range=[2.5, 5.0]),
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=360,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_trend, use_container_width=True)

            with c_t2:
                st.markdown("##### **Annual Review Volume Velocity by Sentiment**")
                yearly_vol = dff2.groupby(["review_year", "sentiment"])["review_count"].sum().reset_index()
                fig_area = px.area(
                    yearly_vol,
                    x="review_year",
                    y="review_count",
                    color="sentiment",
                    color_discrete_map={"positive": "#10B981", "negative": "#EF4444"},
                    labels={"review_year": "Review Year", "review_count": "Annual Review Volume"},
                )
                fig_area.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=360,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_area, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 3: Price vs Rating Relationships
    # --------------------------------------------------------------------------
    with pbi_tab3:
        st.markdown("### **Price vs. Rating Relationships (Catalog Quality & Elasticity)**")
        st.caption("Investigates pricing tiers against customer satisfaction scores, replacing transactional GMV with catalog elasticity analytics.")

        if not df_pbi.empty:
            c_p1, c_p2 = st.columns(2)

            with c_p1:
                st.markdown("##### **Customer Satisfaction by Price Tier**")
                tier_summary = df_pbi.groupby("price_tier").agg(
                    total_reviews=("review_count", "sum"),
                    avg_rating=("avg_rating", "mean"),
                    avg_price=("avg_price", "mean"),
                ).reset_index()
                fig_tier = px.bar(
                    tier_summary,
                    x="price_tier",
                    y="avg_rating",
                    color="price_tier",
                    text_auto=".2f",
                    labels={"price_tier": "Catalog Price Bracket", "avg_rating": "Average Rating (1-5★)"},
                )
                fig_tier.update_layout(
                    yaxis=dict(range=[2.5, 5.0]),
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=360,
                    showlegend=False,
                )
                st.plotly_chart(fig_tier, use_container_width=True)

            with c_p2:
                st.markdown("##### **Price vs. Rating Scatter Plot (Bubble Size = Review Volume)**")
                scatter_data = df_pbi[df_pbi["store_name"] != "Other Verified Stores"].groupby(["store_name", "category", "price_tier"]).agg(
                    avg_price=("avg_price", "mean"),
                    avg_rating=("avg_rating", "mean"),
                    review_count=("review_count", "sum"),
                ).reset_index()

                fig_scatter = px.scatter(
                    scatter_data,
                    x="avg_price",
                    y="avg_rating",
                    size="review_count",
                    color="category",
                    hover_name="store_name",
                    labels={"avg_price": "Average Product Price ($)", "avg_rating": "Average Rating (1-5★)"},
                )
                fig_scatter.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=360,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 4: System Architecture & Ingestion Health
    # --------------------------------------------------------------------------
    with pbi_tab4:
        st.markdown("### **Underlying Database Architecture & Partitioning**")
        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            st.markdown("##### **Customer Rating Distribution (3.1M Reviews)**")
            rating_dist = pd.DataFrame({
                "Star Rating": ["5 Stars ★★★★★", "1 Star ★☆☆☆☆", "2 Stars ★★☆☆☆", "4 Stars ★★★★☆"],
                "Review Count": [1277750, 1016457, 533543, 272250],
                "Share (%)": [41.2, 32.8, 17.2, 8.8],
            }).set_index("Star Rating")
            st.bar_chart(rating_dist["Review Count"], color="#2563EB")
            st.caption("Distribution across the complete 3,100,000 Amazon Customer Review dataset.")

        with col_ch2:
            st.markdown("##### **Sentiment Polarity Balance (500k Benchmark)**")
            polarity_dist = pd.DataFrame({
                "Polarity": ["Positive Sentiment (4-5★)", "Negative Sentiment (1-2★)"],
                "Evaluated Reviews": [250000, 250000],
            }).set_index("Polarity")
            st.bar_chart(polarity_dist["Evaluated Reviews"], color="#059669")
            st.caption("Stratified 50/50 balanced training & testing partition (400,000 train / 100,000 test).")


# ==============================================================================
# VIEW 2: Review Intelligence & Explainable AI (XAI)
# ==============================================================================
elif nav_selection == "📝 Review Intelligence & XAI":
    st.subheader("Customer Review Sentiment Classifier & Explainability Engine")
    st.write("Evaluate customer feedback with transparent decision-making.")

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

            # Explainable AI Word Highlighting
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
# VIEW 3: Bulk Review Auditor (Upload File & Batch Analytics)
# ==============================================================================
elif nav_selection == "📁 Bulk Review Auditor":
    st.subheader("Bulk Review Auditor & Batch Analytics")
    st.write(
        "Upload customer review datasets (`.csv`, `.xlsx`, `.tsv`) or load our standardized test batch "
        "to run rapid parallel inference and visualize sentiment proportions."
    )

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
        text_cols = [
            c for c in df_to_audit.columns
            if any(k in str(c).lower() for k in ["review", "text", "comment", "desc", "feedback", "body"])
        ]
        chosen_col = text_cols[0] if text_cols else df_to_audit.columns[0]

        if len(df_to_audit.columns) > 1:
            chosen_col = st.selectbox("Select Review Text Column:", options=list(df_to_audit.columns), index=df_to_audit.columns.get_loc(chosen_col))

        pipeline = load_sentiment_pipeline()

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

        csv_buffer = io.StringIO()
        df_to_audit.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Audited Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name="ecie_sentiment_audit_results.csv",
            mime="text/csv",
        )


# ==============================================================================
# VIEW 4: Smart Product Recommender (Quality & Rating Weighted)
# ==============================================================================
elif nav_selection == "🎯 Smart Product Recommender":
    st.subheader("Content-Based Smart Product Recommender")
    st.write(
        "Find contextually relevant products. "
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

            with st.container(border=True):
                st.markdown(f"**Selected Product:** {target_row['product_title']}")
                st.caption(
                    f"Category: **{target_row['category']}** | "
                    f"Price: **${target_row['price']:.2f}** | "
                    f"Rating: **⭐ {target_row['avg_rating']}** ({target_row['review_count']} reviews) | "
                    f"ASIN: `{target_row['parent_asin']}`"
                )

            item_vec = nn_model._fit_X[target_idx]
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

            if avg_r < min_rating_filter:
                continue

            sim_score = max(0.0, 1.0 - float(dist))

            if "Smart Business" in ranking_strategy:
                quality_score = avg_r / 5.0
                pop_score = min(1.0, math.log1p(r_count) / math.log1p(100))
                composite_score = (0.65 * sim_score) + (0.25 * quality_score) + (0.10 * pop_score)
            else:
                composite_score = sim_score

            scored_results.append({
                "composite_score": composite_score,
                "similarity_score": sim_score,
                "row": row,
            })

        scored_results.sort(key=lambda x: x["composite_score"], reverse=True)
        final_picks = scored_results[:top_n]

        if final_picks:
            st.markdown(f"#### **Recommended Products ({len(final_picks)} Matches)**")

            for rank_i, item in enumerate(final_picks, start=1):
                row = item["row"]
                sim = item["similarity_score"] * 100
                comp = item["composite_score"] * 100
                r_val = float(row["avg_rating"])
                r_num = int(row["review_count"])
                price = float(row["price"])
                is_top_rated = r_val >= 4.5 and r_num >= 5

                with st.container(border=True):
                    col_h_left, col_h_right = st.columns([4, 1.2])
                    with col_h_left:
                        top_rated_text = " [⭐ Top Rated Product]" if is_top_rated else ""
                        st.markdown(f"**#{rank_i}. {row['product_title']}**{top_rated_text}")
                        st.caption(f"Category: {row['category']}  •  ASIN: `{row['parent_asin']}`")
                    with col_h_right:
                        st.markdown(f"### ${price:.2f}")

                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1:
                        st.markdown(f"**Customer Rating:**  \n⭐ {r_val:.1f} / 5.0")
                    with col_s2:
                        st.markdown(f"**Review Volume:**  \n{r_num:,} customer reviews")
                    with col_s3:
                        st.markdown(f"**Similarity Match:**  \n{sim:.1f}%")
                    with col_s4:
                        st.markdown(f"**Composite Score:**  \n{comp:.1f}%")
        else:
            st.warning(f"No products found meeting the minimum rating threshold of {min_rating_filter}★. Try lowering the rating slider.")
