import yfinance as yf
import pandas as pd
import os

DATA_DIR = r"c:\Users\Sachin D B\Documents\anti_gravity\hedge_fund_v3\architect_v3\data\nse"
TICKER = "^CRSLDX"

def fetch_and_save():
    print(f"Fetching {TICKER}...")
    df = yf.download(TICKER, period="max", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    if df.empty:
        print("Error: No data fetched.")
        return

    # Standardize
    df.index.name = "Date"
    df.reset_index(inplace=True)
    
    # Save
    path = os.path.join(DATA_DIR, f"{TICKER}.parquet")
    df.to_parquet(path)
    print(f"Saved to {path}")

if __name__ == "__main__":
    fetch_and_save()
