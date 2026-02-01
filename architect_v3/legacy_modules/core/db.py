import duckdb
import pandas as pd
import os

class QuantDB:
    def __init__(self, data_path="data/raw/*.parquet"):
        self.data_path = data_path
        # Connect to an in-memory database (Zero latency)
        self.con = duckdb.connect(database=':memory:')
        
    def query(self, sql_query):
        """
        Executes SQL directly on the Parquet files.
        """
        print(f"[DUCKDB] Executing: {sql_query[:50]}...")
        return self.con.execute(sql_query).df()

    def get_universe_by_volume(self, top_n=50, days_lookback=30):
        """
        Selects the most liquid stocks based on average Dollar Volume.
        Crucial for Law 1: Never trade illiquid junk.
        """
        # We use the 'filename' metadata to identify the stock
        # Windows paths might have backslashes, so we handle extraction carefully
        query = fr"""
        WITH recent_data AS (
            SELECT 
                replace(regexp_extract(filename, '([^/\\\\]+)\.parquet$', 1), '.parquet', '') as ticker,
                Close,
                Volume,
                (Close * Volume) as dollar_vol
            FROM read_parquet('{self.data_path}', filename=true)
            WHERE Date >= current_date - INTERVAL '{days_lookback} DAYS'
        )
        SELECT 
            ticker, 
            AVG(dollar_vol) as avg_liquidity 
        FROM recent_data
        GROUP BY ticker
        ORDER BY avg_liquidity DESC
        LIMIT {top_n}
        """
        return self.query(query)

    def get_ticker_data(self, ticker):
        """
        Fast retrieval for a single asset.
        """
        # Ensure we point to the correct file path structure
        # data_path has wildcard, so we can't use it directly here.
        # Construct path from base dir of data_path
        base_dir = os.path.dirname(self.data_path.replace("*.parquet", ""))
        # Handle if data_path was just "*.parquet" -> dirname is empty
        if not base_dir:
            base_dir = "."
            
        file_path = os.path.join(base_dir, f"{ticker}.parquet")
        
        # Windows formatting for SQL? DuckDB handles / usually.
        file_path = file_path.replace("\\", "/")
        
        query = f"""
        SELECT * FROM read_parquet('{file_path}')
        ORDER BY Date ASC
        """
        return self.query(query)

if __name__ == "__main__":
    db = QuantDB()
    
    # 1. Test Universe Selection
    print("Scanning for Liquid Universe...")
    try:
        universe = db.get_universe_by_volume(top_n=10)
        print(universe)
        
        # 2. Test Single Ticker Load
        if not universe.empty:
            top_stock = universe.iloc[0]['ticker']
            print(f"\nLoading Data for Top Asset: {top_stock}")
            df = db.get_ticker_data(top_stock)
            print(df.tail())
    except Exception as e:
        print(f"Error during test: {e}")
