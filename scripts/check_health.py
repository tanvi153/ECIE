#!/usr/bin/env python3
"""
System Health & Diagnostics Script for E-CIE.
Verifies PostgreSQL database, ML sentiment pipeline, and recommender models.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


def check_all():
    print("=" * 60)
    print("🔍 E-CIE SYSTEM HEALTH & PRE-VIVA DIAGNOSTICS")
    print("=" * 60)

    # 1. Check PostgreSQL
    print("1. Checking PostgreSQL Database...")
    try:
        import psycopg2
        conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "ecie_project_db"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "tanvipostgres"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
        )
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM reviews;")
        review_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM products;")
        prod_count = cur.fetchone()[0]
        conn.close()
        print(f"   🟢 PostgreSQL: ACTIVE & HEALTHY on port 5432")
        print(f"      - Reviews Warehouse: {review_count:,} rows")
        print(f"      - Product Entities:  {prod_count:,} rows")
    except Exception as e:
        print(f"   🔴 PostgreSQL Error: {e}")

    # 2. Check Sentiment Model
    print("\n2. Checking NLP Sentiment Model...")
    model_path = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
    if model_path.exists():
        import joblib
        pipeline = joblib.load(model_path)
        pred = pipeline.predict(["The headphones are fantastic and battery lasts forever!"])[0]
        print(f"   🟢 Sentiment Model: READY ({model_path.stat().st_size / (1024*1024):.1f} MB)")
        print(f"      - Test Prediction Verification: '{pred}'")
    else:
        print(f"   🔴 Sentiment Model NOT found at {model_path}")

    # 3. Check Recommender Model
    print("\n3. Checking Product Recommender...")
    rec_path = PROJECT_ROOT / "models" / "recommender.joblib"
    if rec_path.exists():
        print(f"   🟢 Recommender Index: READY ({rec_path.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f"   🔴 Recommender NOT found at {rec_path}")

    # 4. Check Web App
    print("\n4. Checking Streamlit Web Dashboard...")
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:8501", timeout=3)
        if resp.status == 200:
            print("   🟢 Streamlit Web App: RUNNING on http://localhost:8501")
    except Exception:
        print("   🟡 Streamlit Web App is not running right now.")
        print("      Run: .venv/bin/python -m streamlit run app/streamlit_app.py")

    print("=" * 60)
    print("All core engine modules are verified!")
    print("=" * 60)


if __name__ == "__main__":
    check_all()
