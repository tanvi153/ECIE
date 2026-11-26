#!/usr/bin/env python3
"""
Export analytical views from PostgreSQL to CSV files for Power BI import.
Generates:
  - data/powerbi/category_store_sentiment.csv
  - data/powerbi/rating_trends_monthly.csv
  - data/powerbi/price_vs_rating_catalog.csv
"""

import os
import sys
from pathlib import Path
import psycopg2
import pandas as pd
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / "data" / "powerbi"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "ecie_project_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "tanvipostgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )


def export_powerbi_data():
    conn = get_db_connection()
    print("Connected to PostgreSQL database:", os.getenv("POSTGRES_DB", "ecie_project_db"))

    # 1. Category & Store Sentiment
    print("Exporting v_powerbi_category_store...")
    df_cat_store = pd.read_sql("SELECT * FROM v_powerbi_category_store ORDER BY category, store_name, sentiment", conn)
    cat_store_path = OUT_DIR / "category_store_sentiment.csv"
    df_cat_store.to_csv(cat_store_path, index=False)
    print(f" Saved {len(df_cat_store):,} rows to {cat_store_path}")

    # 2. Rating Trends Monthly
    print("Exporting v_powerbi_rating_trends...")
    df_trends = pd.read_sql("SELECT * FROM v_powerbi_rating_trends ORDER BY review_month_date, category, sentiment", conn)
    trends_path = OUT_DIR / "rating_trends_monthly.csv"
    df_trends.to_csv(trends_path, index=False)
    print(f" Saved {len(df_trends):,} rows to {trends_path}")

    # 3. Price vs Rating Catalog (Top 25,000 products by review count for ultra-responsive Power BI scatter plots)
    print("Exporting v_powerbi_price_vs_rating (top 25k products with >= 5 reviews)...")
    df_price_rating = pd.read_sql(
        """
        SELECT * 
        FROM v_powerbi_price_vs_rating 
        WHERE total_reviews >= 5
        ORDER BY total_reviews DESC 
        LIMIT 25000
        """,
        conn,
    )
    price_rating_path = OUT_DIR / "price_vs_rating_catalog.csv"
    df_price_rating.to_csv(price_rating_path, index=False)
    print(f" Saved {len(df_price_rating):,} rows to {price_rating_path}")

    # 4. Master Unified Report (All-in-One for Power BI Web pasting)
    print("Exporting master_unified_report (all-in-one table for Power BI Web paste)...")
    df_master = pd.read_sql(
        """
        SELECT 
            r.category,
            CASE 
                WHEN s.store_name IN (
                    'Anker Direct', 'Sony Official', 'Samsung Electronics', 'Logitech Store', 
                    'Apple Authorized', 'Bose Sound', 'Spigen Official', 'Amazon Basics', 
                    'Belkin Tech', 'Corsair Gaming', 'Sennheiser Pro', 'ASUS ROG Store', 
                    'Razer Official', 'JBL Audio Store', 'Dell Technologies', 'HP Official', 
                    'Lenovo Authorized', 'TP-Link Networking', 'Netgear Official', 'SanDisk Store', 
                    'Seagate Storage', 'Western Digital', 'Kingston Tech', 'Crucial Memory', 
                    'Garmin Fitness'
                ) THEN s.store_name
                ELSE 'Other Verified Stores'
            END AS store_name,
            EXTRACT(YEAR FROM TO_TIMESTAMP(r.timestamp::numeric / 1000.0))::integer AS review_year,
            CASE 
                WHEN p.price < 25.00 THEN '1. Budget (under $25)'
                WHEN p.price < 75.00 THEN '2. Mid-Range ($25-$75)'
                WHEN p.price < 150.00 THEN '3. Upper Mid ($75-$150)'
                ELSE '4. Premium ($150+)'
            END AS price_tier,
            ROUND(AVG(p.price), 2) AS avg_price,
            r.sentiment,
            COUNT(*) AS review_count,
            ROUND(AVG(r.rating), 2) AS avg_rating,
            COUNT(CASE WHEN r.verified_purchase = true THEN 1 END) AS verified_reviews
        FROM reviews r
        JOIN products p ON r.parent_asin = p.parent_asin
        JOIN stores s ON p.store_id = s.store_id
        WHERE r.timestamp IS NOT NULL
          AND EXTRACT(YEAR FROM TO_TIMESTAMP(r.timestamp::numeric / 1000.0)) BETWEEN 2012 AND 2023
        GROUP BY 
            1, 2, 3, 4, 6
        ORDER BY review_count DESC;
        """,
        conn,
    )
    master_path = OUT_DIR / "master_unified_report.csv"
    df_master.to_csv(master_path, index=False)
    print(f" Saved {len(df_master):,} rows to {master_path}")

    # Also save as Tab-Separated (TSV) for direct clipboard paste into Power BI Web grid
    tsv_path = OUT_DIR / "master_unified_report.tsv"
    df_master.to_csv(tsv_path, sep="\t", index=False)
    print(f" Saved {len(df_master):,} rows to {tsv_path} (Tab-Separated for Power BI Web paste)")

    conn.close()
    print("All Power BI datasets successfully exported!")


if __name__ == "__main__":
    export_powerbi_data()
