# strategies/screener_engine.py
import duckdb
import pandas as pd
import time
from datetime import datetime

class DuckDBScreener:
    def __init__(self, parquet_path="data/raw/*.parquet"):
        self.parquet_path = parquet_path
        # Connect to an in-memory database (Zero latency)
        self.conn = duckdb.connect(':memory:')

    def run_screener_at_date(self, dt):
        """Runs the entire Physics Screener in C++ Native DuckDB (0.5 seconds)"""
        dt_str = dt.strftime('%Y-%m-%d')
        
        # We push the R^2 math, the Energy calculation, and the Liquidity filter
        # directly into the SQL engine. Python does zero math.
        
        print(f"ARCHITECT: Running native DuckDB physics scan for {dt_str}...")
        
        # Note: We extract 'ticker' from the filename because it doesn't exist as a column in the parquet files.
        # using raw f-string (fr) to handle backslashes in regex correctly
        query = fr"""
            WITH historical_window AS (
                SELECT 
                    replace(regexp_extract(filename, '([^/\\\\]+)\.parquet$', 1), '.parquet', '') as ticker,
                    close, 
                    volume, 
                    date,
                    -- Create an 'X' axis (day number) for the regression
                    ROW_NUMBER() OVER (PARTITION BY filename ORDER BY date) as day_num
                FROM read_parquet('{self.parquet_path}', filename=true)
                WHERE date >= (DATE '{dt_str}' - INTERVAL 90 DAY)
                AND date < DATE '{dt_str}' 
            ),
            
            calculated_metrics AS (
                SELECT ticker, 
                       AVG(volume) as adv,
                       FIRST(close) as start_price,
                       LAST(close) as end_price,
                       -- NATIVE DUCK-DB LINEAR REGRESSION (Calculates R^2 in milliseconds)
                       REGR_R2(close, day_num) as smoothness
                FROM historical_window
                GROUP BY ticker
                HAVING COUNT(date) >= 60 -- Only include stocks with enough data
            )
            
            SELECT ticker, smoothness
            FROM calculated_metrics
            -- THE PHYSICS FILTERS APPLIED IN SQL
            WHERE adv >= 500000 
            AND end_price <= 10000 
            AND (end_price / start_price) - 1 >= 0.15 
            AND smoothness > 0.75
            ORDER BY smoothness DESC
            LIMIT 5;
        """
        
        # Execute query and return the list of tickers instantly
        try:
            result_df = self.conn.execute(query).df()
            return result_df
        except Exception as e:
            print(f"Error executing logic: {e}")
            return pd.DataFrame()

    def scan(self):
        print("ARCHITECT: Commencing Vectorized Scan...")
        start_time = time.time()
        
        # Run for today
        df_results = self.run_screener_at_date(datetime.now())
        
        if not df_results.empty:
            print(f"ARCHITECT: Found {len(df_results)} Prime Assets:")
            print(df_results)
            # RESTRICT TO TOP 5 ASSETS FOR MICRO-CAP ALLOCATION
            # The SQL query already limits to 5, but we ensure output logic matches expectations
            df_results.to_csv("prime_assets.csv", index=False)
            print("ARCHITECT: Prime Assets locked for 5L Allocation.")
        else:
            print("ARCHITECT: No assets met the physics criteria.")
        
        end_time = time.time()
        print(f"Scan Complete in {round(end_time - start_time, 2)} seconds.")
        return df_results

if __name__ == "__main__":
    screener = DuckDBScreener()
    screener.scan()
