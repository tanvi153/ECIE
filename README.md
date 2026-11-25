# ECIE — E-Commerce Intelligence Engine

An end-to-end data engineering and machine learning pipeline for e-commerce sentiment analysis. ECIE provides structured PostgreSQL data ingestion, automated review text cleaning, high-capacity TF-IDF feature extraction, ultra-high-precision SGD linear classification, and experiment tracking powered by MLflow.

---

## 📁 Project Architecture & Tree

```text
ECIE/
├── data/
│   ├── raw/                       # Place raw CSV files here (table1_reviews.csv, etc.)
│   │   └── .gitkeep
│   └── processed/                 # Processed datasets and intermediate tables
│       └── .gitkeep
├── experiments/                   # Experiment tracking database & metrics
│   └── mlflow.db                  # MLflow SQLite backend database
├── models/                        # Pre-trained models and serialized vectorizers
│   ├── sentiment_model.joblib     # High-precision SGDClassifier (Modified Huber)
│   └── tfidf_vectorizer.joblib    # 3-gram TF-IDF Vectorizer (300K features)
├── notebooks/                     # Exploratory Data Analysis & experiments
│   └── main.ipynb                 # EDA, dataset schema inspection, & summary
├── sql/                           # Database schemas, table definitions, & SQL views
│   └── schema.sql                 # Table schemas and 'v_cleaned_reviews' view definition
├── src/                           # Production source code
│   ├── __init__.py
│   ├── config.py                  # Centralized configuration & environment loader
│   ├── database.py                # Reusable PostgreSQL engine & connection tester
│   ├── data/                      # Data pipeline & ingestion
│   │   ├── __init__.py
│   │   └── ingest.py              # Chunked ingestion pipeline for all 5 CSV tables
│   └── models/                    # ML pipeline & inference
│       ├── __init__.py
│       ├── train.py               # Training pipeline with MLflow logging
│       └── predict.py             # CLI & programmatic sentiment inference
├── .env.example                   # Environment configuration template
├── .gitattributes                 # Git LFS rules for large binaries & datasets
├── .gitignore                     # Git ignore rules for datasets, cache, & credentials
├── requirements.txt               # Pinned Python package dependencies
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- PostgreSQL server (running locally or remotely)

### 2. Environment Setup

Clone the repository and set up a virtual environment:

```bash
cd /home/aryan/Documents/ECIE

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Database Credentials

Copy the environment template and customize your settings:

```bash
cp .env.example .env
```

Open `.env` and set your PostgreSQL credentials:

```env
DB_USER=postgres
DB_PASS=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ecie_project_db
```

---

## 🗄️ Database Schema & Analytical Views

Initialize the database tables and analytical views using `sql/schema.sql`:

```bash
psql -U postgres -d ecie_project_db -f sql/schema.sql
```

The database manages 5 core e-commerce relational entities:
- **`stores`**: Merchant and store metadata.
- **`users`**: Customer profiles and signup history.
- **`products`**: Catalog details, prices, and merchant associations.
- **`reviews`**: Review metadata, star ratings, and verified purchase flags.
- **`review_text`**: Review headline and review body content.
- **`v_cleaned_reviews`**: SQL view combining ratings and text with mutual-exclusion polarity labeling.

---

## 📥 Data Ingestion

Place the source CSV files into `data/raw/`:
- `table1_reviews.csv`
- `table2_review_text.csv`
- `table3_products.csv`
- `table4_stores.csv`
- `table5_users.csv`

Run the automated ingestion pipeline:

```bash
# Ingest all 5 datasets in dependency order:
python -m src.data.ingest --all

# Or ingest a specific dataset:
python -m src.data.ingest --table reviews
python -m src.data.ingest --table products --file /path/to/custom_products.csv
```

---

## 🧠 Model Training & Experiment Tracking

The training pipeline:
1. Fetches high-certainty review samples from PostgreSQL (`v_cleaned_reviews`).
2. Performs contraction expansion (e.g. *don't* &rarr; *do not*).
3. Fits a high-capacity 1-to-3 N-gram TF-IDF vectorizer (300,000 features).
4. Trains an averaged `SGDClassifier` with Modified Huber loss for calibrated probabilities.
5. Logs parameters, classification reports, and metrics to MLflow (`experiments/mlflow.db`).
6. Serializes artifacts to `models/sentiment_model.joblib` and `models/tfidf_vectorizer.joblib`.

Execute training:

```bash
python -m src.models.train --limit 250000
```

### Inspect Experiments in MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db
```
Navigate to `http://localhost:5000` to review runs, metrics, and parameters.

---

## 🔮 Sentiment Inference

### Via Command-Line Interface (CLI):

```bash
python -m src.models.predict "The product arrived on time and exceeded all my expectations!"
```

**Output:**
```text
--- Sentiment Prediction ---
Review:     The product arrived on time and exceeded all my expectations!
Sentiment:  POSITIVE
Confidence: 99.82%
Probabilities: {'negative': 0.0018, 'positive': 0.9982}
```

### In Python Code:

```python
from src.models import predict_sentiment

result = predict_sentiment("Defective item, broke after two days. Do not buy.")
print(result["sentiment"])   # 'negative'
print(result["confidence"])  # 0.994...
```

---

## 📊 Exploratory Data Analysis (EDA)

The exploratory notebook is available at [`notebooks/main.ipynb`](notebooks/main.ipynb). It automatically detects dataset locations and produces schema, shape, data type, and memory statistics for all 5 relational tables.
