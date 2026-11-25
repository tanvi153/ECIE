"""
Data Loading Module for ECIE.
Provides utilities and CLI scripts to load CSV tables into PostgreSQL.
"""

from src.data.ingest import load_all_tables, load_table, resolve_csv_path

__all__ = ["load_table", "load_all_tables", "resolve_csv_path"]
