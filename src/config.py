"""
Centralized Configuration Module for ECIE.

Manages directory paths, database connection parameters, and MLflow tracking
settings with fallback defaults and environment variable support (.env).
"""

import os
from pathlib import Path

# Load environment variables from .env if python-dotenv is installed
try:
    from dotenv import load_dotenv

    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass

# ==============================================================================
# 1. Project Directory Paths
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", DATA_DIR / "raw"))
DATA_PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", DATA_DIR / "processed"))

MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SQL_DIR = PROJECT_ROOT / "sql"

# Ensure runtime directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Model Artifact Paths
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
SENTIMENT_PIPELINE_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
RECOMMENDER_MODEL_PATH = MODELS_DIR / "recommender.joblib"
PRODUCTS_SAMPLE_CSV_PATH = DATA_DIR / "products_sample.csv"

# ==============================================================================
# 2. Database Configuration
# ==============================================================================
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ecie_project_db")

_db_auth = f"{DB_USER}:{DB_PASS}" if DB_PASS else DB_USER
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+psycopg2://{_db_auth}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)

# ==============================================================================
# 3. MLflow Tracking Configuration
# ==============================================================================
MLFLOW_DB_PATH = EXPERIMENTS_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}",
)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ECIE_Sentiment_Analysis")
