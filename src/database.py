"""
Database Connection Utility for ECIE.

Provides centralized SQLAlchemy engine creation and connectivity checks
for PostgreSQL database operations.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import DATABASE_URL


def get_db_engine(echo: bool = False) -> Engine:
    """
    Creates and returns a SQLAlchemy Engine for the ECIE PostgreSQL database.

    Args:
        echo (bool): Whether to log SQL statements.

    Returns:
        Engine: SQLAlchemy Engine connected to DATABASE_URL.
    """
    return create_engine(DATABASE_URL, echo=echo)


def test_db_connection() -> bool:
    """
    Tests the PostgreSQL database connection.

    Returns:
        bool: True if connection is successful, False otherwise.
    """
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1;")).scalar()
            return result == 1
    except Exception as exc:
        print(f"[!] Database connection test failed: {exc}")
        return False
