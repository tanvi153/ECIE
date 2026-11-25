"""
Dataset Loading Script for ECIE.

Safely loads raw CSV files into PostgreSQL tables without hardcoded credentials.
Uses environment variables configured in .env via src.config.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingest import TABLE_MAPPINGS, load_all_tables, load_table


def main():
    parser = argparse.ArgumentParser(
        description="Securely ingest CSV datasets into PostgreSQL tables without hardcoded credentials."
    )
    parser.add_argument(
        "--table",
        type=str,
        choices=list(TABLE_MAPPINGS.keys()),
        help="Specific table to load (reviews, review_text, products, stores, users)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Explicit path to source CSV file (optional).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing the CSV tables (default: data/raw).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sequentially load all 5 relational tables.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None

    if args.all:
        load_all_tables(data_dir=data_dir)
    elif args.table:
        load_table(table_name=args.table, csv_path=args.file, data_dir=data_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
