"""
High-Speed Bulk Ingestion for 3.1M Amazon Reviews.
Uses PostgreSQL's native COPY command via psycopg2 copy_expert for maximum throughput.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_RAW_DIR
from src.database import get_db_engine


def fast_load_table1_reviews(csv_path: Path):
    print(f"\n[1/2] Loading {csv_path.name} into 'reviews' via COPY...")
    t0 = time.time()
    engine = get_db_engine()
    raw_conn = engine.raw_connection()
    cur = raw_conn.cursor()

    # Truncate existing reviews to allow clean bulk reload
    print("  Truncating existing 'reviews' table...")
    cur.execute("TRUNCATE TABLE reviews CASCADE;")
    raw_conn.commit()

    # Drop indexes temporarily for maximum COPY throughput
    print("  Optimizing table for bulk load...")
    cur.execute("""
        DROP INDEX IF EXISTS idx_reviews_rating;
        DROP INDEX IF EXISTS idx_reviews_sentiment;
        DROP INDEX IF EXISTS idx_reviews_parent_asin;
        DROP INDEX IF EXISTS idx_reviews_user_id;
    """)
    raw_conn.commit()

    sql_copy = """
        COPY reviews (review_id, parent_asin, user_id, rating, sentiment, timestamp, helpful_vote, verified_purchase, category)
        FROM STDIN WITH (FORMAT csv, HEADER true);
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        cur.copy_expert(sql_copy, f)
    raw_conn.commit()
    t_load = time.time() - t0

    cur.execute("SELECT count(*) FROM reviews;")
    count = cur.fetchone()[0]
    print(f"  [✓] Loaded {count:,} rows into 'reviews' in {t_load:.1f}s ({count / max(0.1, t_load):,.0f} rows/s)!")

    # Recreate performance indexes
    print("  Rebuilding performance indexes...")
    t_idx = time.time()
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
        CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment);
        CREATE INDEX IF NOT EXISTS idx_reviews_parent_asin ON reviews(parent_asin);
        CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON reviews(user_id);
    """)
    raw_conn.commit()
    print(f"  [✓] Indexes rebuilt in {time.time() - t_idx:.1f}s")

    cur.close()
    raw_conn.close()


class NullByteFilter:
    """Wraps a text stream to strip null bytes (\\x00) which PostgreSQL disallows."""
    def __init__(self, stream):
        self.stream = stream

    def read(self, size=-1):
        chunk = self.stream.read(size)
        return chunk.replace("\0", "") if chunk else ""

    def readline(self, size=-1):
        line = self.stream.readline(size)
        return line.replace("\0", "") if line else ""


def fast_load_table2_review_text(csv_path: Path):
    print(f"\n[2/2] Loading {csv_path.name} into 'review_text' via COPY...")
    t0 = time.time()
    engine = get_db_engine()
    raw_conn = engine.raw_connection()
    cur = raw_conn.cursor()

    print("  Truncating existing 'review_text' table...")
    cur.execute("TRUNCATE TABLE review_text;")
    raw_conn.commit()

    print("  Optimizing table for bulk load...")
    cur.execute("""
        ALTER TABLE review_text DROP CONSTRAINT IF EXISTS review_text_pkey;
        DROP INDEX IF EXISTS idx_review_text_review_id;
    """)
    raw_conn.commit()

    sql_copy = """
        COPY review_text (review_id, review_title, review_text)
        FROM STDIN WITH (FORMAT csv, HEADER true);
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        cur.copy_expert(sql_copy, NullByteFilter(f))
    raw_conn.commit()
    t_load = time.time() - t0

    cur.execute("SELECT count(*) FROM review_text;")
    count = cur.fetchone()[0]
    print(f"  [✓] Loaded {count:,} rows into 'review_text' in {t_load:.1f}s ({count / max(0.1, t_load):,.0f} rows/s)!")

    print("  Rebuilding primary key on review_id...")
    t_idx = time.time()
    cur.execute("""
        ALTER TABLE review_text ADD CONSTRAINT review_text_pkey PRIMARY KEY (review_id);
    """)
    raw_conn.commit()
    print(f"  [✓] Primary key rebuilt in {time.time() - t_idx:.1f}s")

    cur.close()
    raw_conn.close()


def main():
    p1 = DATA_RAW_DIR / "table1_reviews.csv"
    p2 = DATA_RAW_DIR / "table2_review_text.csv"

    if not p1.exists():
        raise FileNotFoundError(f"Missing {p1}")
    if not p2.exists():
        raise FileNotFoundError(f"Missing {p2}")

    t_all = time.time()
    fast_load_table1_reviews(p1)
    fast_load_table2_review_text(p2)
    print(f"\nAll 3.1 Million reviews successfully ingested in {time.time() - t_all:.1f}s!")


if __name__ == "__main__":
    main()
