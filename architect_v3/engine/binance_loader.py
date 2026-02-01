import requests
import zipfile
import io
import polars as pl
import os
from tqdm import tqdm

# CONFIGURATION
SYMBOL = "ETHUSDT"
YEAR = "2023"
MONTH = "01"  # January
DATA_TYPE = "trades" # Options: 'trades', 'aggTrades'
BASE_URL = f"https://data.binance.vision/data/spot/monthly/{DATA_TYPE}/{SYMBOL}/"
FILENAME = f"{SYMBOL}-{DATA_TYPE}-{YEAR}-{MONTH}"
URL = f"{BASE_URL}{FILENAME}.zip"
OUTPUT_DIR = "data/raw_crypto"
OUTPUT_PARQUET = f"{OUTPUT_DIR}/{FILENAME}.parquet"

def fetch_and_convert():
    print(f"[ARCHITECT] Initiating Vector Retrieval: {URL}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Stream Download
    response = requests.get(URL, stream=True)
    if response.status_code != 200:
        print(f"[ERROR] Failed to retrieve data. Status: {response.status_code}")
        return

    # 2. In-Memory Unzip (Avoid writing huge CSVs to disk)
    total_size = int(response.headers.get('content-length', 0))
    
    with io.BytesIO() as buffer:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                buffer.write(chunk)
                pbar.update(len(chunk))
        
        print("[ARCHITECT] Decompressing stream...")
        with zipfile.ZipFile(buffer) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                
                # 3. Polars Parsing (Zero-Copy)
                # Binance Trade Columns: id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch
                print(f"[ARCHITECT] Parsing CSV stream to Polars LazyFrame...")
                
                # We define schema manually for speed/safety
                schema = {
                    "id": pl.Int64,
                    "price": pl.Float64,
                    "qty": pl.Float64,
                    "quote_qty": pl.Float64,
                    "time": pl.Int64, # Unix ms
                    "is_buyer_maker": pl.Boolean,
                    "is_best_match": pl.Boolean
                }
                
                df = pl.read_csv(f.read(), has_header=False, new_columns=list(schema.keys()))
                
                # 4. Format Optimization
                df = df.with_columns([
                    (pl.col("time") * 1000).cast(pl.Datetime("us")).alias("datetime")
                ])
                
                # 5. Write to Parquet
                print(f"[ARCHITECT] Compressing to Parquet: {OUTPUT_PARQUET}")
                df.write_parquet(OUTPUT_PARQUET, compression="snappy")
                
    print(f"[SUCCESS] High-Frequency Vector Ready: {os.path.getsize(OUTPUT_PARQUET) / 1e6:.2f} MB")

if __name__ == "__main__":
    fetch_and_convert()
