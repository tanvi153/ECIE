"""
Git LFS Downloader for ECIE.
Streams large dataset CSVs directly from GitHub Git LFS media storage into data/raw/.
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_RAW_DIR

LFS_BATCH_URL = "https://github.com/Ta20-ops/ecommerce_csvs.git/info/lfs/objects/batch"

LFS_OBJECTS = {
    "table1_reviews.csv": {
        "oid": "6358f94b77ab630d7a1e49a129fafc76bc26e2e3ec0d8c925910923a57a324d2",
        "size": 314020153,
    },
    "table2_review_text.csv": {
        "oid": "9f2578ccdd509d689a04edd1c63e88472b9635ab8e70a055bbbc728c17613b9c",
        "size": 973435198,
    },
}


def get_lfs_download_urls():
    headers = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
        "User-Agent": "git-lfs/3.0.0",
    }
    payload = {
        "operation": "download",
        "transfers": ["basic"],
        "objects": [
            {"oid": meta["oid"], "size": meta["size"]}
            for meta in LFS_OBJECTS.values()
        ],
    }

    req = urllib.request.Request(
        LFS_BATCH_URL, data=json.dumps(payload).encode("utf-8"), headers=headers
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    url_map = {}
    oid_to_filename = {v["oid"]: k for k, v in LFS_OBJECTS.items()}
    for obj in data.get("objects", []):
        oid = obj.get("oid")
        filename = oid_to_filename.get(oid)
        download_url = obj.get("actions", {}).get("download", {}).get("href")
        if filename and download_url:
            url_map[filename] = download_url

    return url_map


def download_file(url: str, dest_path: Path, expected_size: int):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")

    if dest_path.exists() and dest_path.stat().st_size == expected_size:
        print(f"  [✓] {dest_path.name} already exists ({expected_size / (1024**2):.1f} MB). Skipping.")
        return

    print(f"  Downloading {dest_path.name} ({expected_size / (1024**2):.1f} MB)...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    t0 = time.time()
    downloaded = 0
    chunk_size = 1024 * 1024 * 4  # 4 MB chunks

    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - t0
            speed = (downloaded / (1024**2)) / max(0.1, elapsed)
            percent = (downloaded / expected_size) * 100
            sys.stdout.write(f"\r      {downloaded/(1024**2):.1f} MB / {expected_size/(1024**2):.1f} MB ({percent:.1f}%) @ {speed:.1f} MB/s")
            sys.stdout.flush()

    print()
    tmp_path.replace(dest_path)
    print(f"  [✓] Downloaded {dest_path.name} in {time.time()-t0:.1f}s")


def main():
    print("=== [Step 1/3] Fetching Git LFS Download URLs ===")
    url_map = get_lfs_download_urls()
    print(f"Obtained download URLs for {len(url_map)} files.")

    for filename, meta in LFS_OBJECTS.items():
        dest = DATA_RAW_DIR / filename
        url = url_map[filename]
        download_file(url, dest, meta["size"])

    print("\nAll raw CSV files successfully downloaded to data/raw/!")


if __name__ == "__main__":
    main()
