"""
Data Ingestion Pipeline for ECIE.

Streams and loads CSV datasets into PostgreSQL tables using chunked batch insertion:
- table1_reviews.csv      -> reviews
- table2_review_text.csv  -> review_text
- table3_products.csv     -> products
- table4_stores.csv       -> stores
- table5_users.csv        -> users
"""

import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
from sqlalchemy.engine import Engine

from src.config import DATA_RAW_DIR
from src.database import get_db_engine

# Mapping of database table names to default CSV file names and ingestion settings
TABLE_MAPPINGS = {
    "reviews": {
        "filename": "table1_reviews.csv",
        "chunked": True,
        "read_chunksize": 50000,
        "insert_chunksize": 10000,
    },
    "review_text": {
        "filename": "table2_review_text.csv",
        "chunked": True,
        "read_chunksize": 50000,
        "insert_chunksize": 10000,
    },
    "products": {
        "filename": "table3_products.csv",
        "chunked": False,
        "read_chunksize": None,
        "insert_chunksize": 10000,
    },
    "stores": {
        "filename": "table4_stores.csv",
        "chunked": False,
        "read_chunksize": None,
        "insert_chunksize": 10000,
    },
    "users": {
        "filename": "table5_users.csv",
        "chunked": False,
        "read_chunksize": None,
        "insert_chunksize": 10000,
    },
}


def resolve_csv_path(table_name: str, custom_path: Optional[str] = None, data_dir: Optional[Path] = None) -> Path:
    """
    Resolves the exact CSV file path for a given table.
    """
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified CSV file not found: {custom_path}")

    search_dir = Path(data_dir) if data_dir else DATA_RAW_DIR
    target_filename = TABLE_MAPPINGS[table_name]["filename"]

    # Check search directory (data/raw) or custom data_dir
    primary_path = search_dir / target_filename
    if primary_path.exists():
        return primary_path

    # Check relative ecommerce_csvs folder if present
    relative_csv_path = search_dir.parent / "ecommerce_csvs" / target_filename
    if relative_csv_path.exists():
        return relative_csv_path

    raise FileNotFoundError(
        f"Could not find '{target_filename}' in '{search_dir}'. "
        f"Please place the CSV file in '{DATA_RAW_DIR}' or provide --file."
    )


def load_table(
    table_name: str,
    csv_path: Optional[str] = None,
    data_dir: Optional[Path] = None,
    engine: Optional[Engine] = None,
    if_exists: str = "append",
) -> None:
    """
    Loads a specific CSV file into the target PostgreSQL table.

    Args:
        table_name (str): One of ('reviews', 'review_text', 'products', 'stores', 'users').
        csv_path (Optional[str]): Explicit path to CSV. If omitted, resolved automatically.
        data_dir (Optional[Path]): Directory containing raw CSV files.
        engine (Optional[Engine]): SQLAlchemy engine.
        if_exists (str): Mode for table insertion ('append', 'replace', 'fail').
    """
    if table_name not in TABLE_MAPPINGS:
        raise ValueError(
            f"Unknown table '{table_name}'. Valid options: {list(TABLE_MAPPINGS.keys())}"
        )

    file_path = resolve_csv_path(table_name, custom_path=csv_path, data_dir=data_dir)
    db_engine = engine or get_db_engine()
    config = TABLE_MAPPINGS[table_name]

    print(f"\n[+] Ingesting '{table_name}' from: {file_path}")

    if config["chunked"]:
        chunk_num = 1
        total_rows = 0
        read_chunksize = config["read_chunksize"]
        insert_chunksize = config["insert_chunksize"]

        for chunk in pd.read_csv(file_path, chunksize=read_chunksize):
            chunk.to_sql(
                name=table_name,
                con=db_engine,
                if_exists=if_exists if chunk_num == 1 else "append",
                index=False,
                method="multi",
                chunksize=insert_chunksize,
            )
            total_rows += len(chunk)
            print(f"    Chunk {chunk_num} ({len(chunk)} rows) inserted... (Total so far: {total_rows})")
            chunk_num += 1

        print(f"[✓] Successfully loaded {total_rows} rows into '{table_name}' table!")
    else:
        df = pd.read_csv(file_path)
        print(f"    Read {len(df)} rows from CSV. Inserting into '{table_name}'...")
        df.to_sql(
            name=table_name,
            con=db_engine,
            if_exists=if_exists,
            index=False,
            method="multi",
            chunksize=config["insert_chunksize"],
        )
        print(f"[✓] Successfully loaded {len(df)} rows into '{table_name}' table!")


def load_all_tables(
    data_dir: Optional[Path] = None,
    engine: Optional[Engine] = None,
    if_exists: str = "append",
) -> None:
    """
    Loads all 5 tables in order into PostgreSQL.
    """
    db_engine = engine or get_db_engine()
    ordered_tables = ["stores", "users", "products", "reviews", "review_text"]

    for table in ordered_tables:
        try:
            load_table(table, data_dir=data_dir, engine=db_engine, if_exists=if_exists)
        except FileNotFoundError as err:
            print(f"[!] Warning: Skipping '{table}' - {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ECIE PostgreSQL Data Ingestion Utility")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load all 5 datasets in order (stores, users, products, reviews, review_text)",
    )
    parser.add_argument(
        "--table",
        choices=list(TABLE_MAPPINGS.keys()),
        help="Specific table name to load",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional explicit path to the CSV file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory where CSV files are located (defaults to data/raw)",
    )
    parser.add_argument(
        "--if-exists",
        choices=["append", "replace", "fail"],
        default="append",
        help="Insertion mode if table exists (default: append)",
    )

    args = parser.parse_args()

    custom_dir = Path(args.data_dir) if args.data_dir else None

    if args.all:
        load_all_tables(data_dir=custom_dir, if_exists=args.if_exists)
    elif args.table:
        load_table(
            args.table,
            csv_path=args.file,
            data_dir=custom_dir,
            if_exists=args.if_exists,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
