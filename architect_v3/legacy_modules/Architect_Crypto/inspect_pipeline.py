# Architect_Crypto/inspect_pipeline.py
import pandas as pd
import duckdb
import os

DATA_DIR = "crypto_data"

def inspect_parquet_structure():
    print("ARCHITECT: Rigorous Data Pipeline Inspection")
    print("-" * 50)
    
    if not os.path.exists(DATA_DIR):
        print(f"CRITICAL ERROR: {DATA_DIR} does not exist.")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')]
    conn = duckdb.connect(':memory:')

    for f in files:
        path = os.path.join(DATA_DIR, f)
        print(f"\nInspecting: {f}")
        
        # 1. Load Data
        df = pd.read_parquet(path)
        
        # 2. Check Columns
        expected_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        cols = list(df.columns)
        print(f"Columns: {cols}")
        if not all(col in cols for col in expected_cols):
             print(f"FAIL: Missing columns. Expected {expected_cols}")
        else:
             print("PASS: Column structure valid.")
             
        # 3. Check Timestamp Format & Timezone
        first_date = df['date'].iloc[0]
        last_date = df['date'].iloc[-1]
        dtype = df['date'].dtype
        print(f"Date Dtype: {dtype}")
        print(f"Range: {first_date} to {last_date}")
        
        if hasattr(first_date, 'tzinfo') and first_date.tzinfo is not None:
             print(f"WARN: Timezone Detected: {first_date.tzinfo}")
        else:
             print("PASS: Timestamps are Naive (No Timezone).")
             
        # 4. Check Ordering
        if df['date'].is_monotonic_increasing:
             print("PASS: Data is strictly ordered.")
        else:
             print("FAIL: Data is NOT ordered.")
             
        # 5. Check Head/Tail values
        print("Head:")
        print(df.head(3)[['date', 'close', 'volume']])
        
    conn.close()

if __name__ == "__main__":
    inspect_parquet_structure()
