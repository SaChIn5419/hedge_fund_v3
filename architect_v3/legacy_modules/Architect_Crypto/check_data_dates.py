# Architect_Crypto/check_data_dates.py
import duckdb
import pandas as pd
import os

DATA_DIR = "crypto_data"

def check_dates():
    print(f"ARCHITECT: Checking Data Ranges in {DATA_DIR}...")
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: {DATA_DIR} does not exist.")
        return

    conn = duckdb.connect(':memory:')
    
    # Get all parquet files
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')]
    
    for f in files:
        path = os.path.join(DATA_DIR, f)
        query = f"SELECT MIN(date) as start_date, MAX(date) as end_date, COUNT(*) as count FROM '{path}'"
        try:
            df = conn.execute(query).df()
            print(f"\nAsset: {f}")
            print(df)
            
            if "SOL" in f:
                print("SOL Data Head:")
                q_head = f"SELECT * FROM '{path}' ORDER BY date ASC LIMIT 10"
                print(conn.execute(q_head).df())
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    conn.close()

if __name__ == "__main__":
    check_dates()
