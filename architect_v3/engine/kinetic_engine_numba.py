import polars as pl
import numpy as np
import duckdb
from numba import jit
import os
import sys

# Windows Console Encoding Fix
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ==========================================
# 1. NUMBA KERNELS (The Mathematics)
# ==========================================
# We compile these to machine code for C-speed execution.

@jit(nopython=True, nogil=True, cache=True)
def _numba_rolling_fdi(price_array, window):
    """
    Sevcik Fractal Dimension Index on rolling window.
    Complexity: O(N * W)
    """
    n = len(price_array)
    output = np.full(n, np.nan, dtype=np.float64)
    
    # Pre-compute constants
    log_2 = np.log(2.0)
    log_2_n_minus_1 = np.log(2.0 * (window - 1.0))
    dt_sq = (1.0 / (window - 1.0)) ** 2
    
    for i in range(window, n):
        # Window slice
        chunk = price_array[i-window:i]
        
        max_p = np.max(chunk)
        min_p = np.min(chunk)
        range_p = max_p - min_p
        
        if range_p <= 1e-9: # Avoid division by zero
            output[i] = 1.0 
            continue
            
        # Variogram loop (Length calculation)
        length = 0.0
        # Normalize first point
        prev_norm = (chunk[0] - min_p) / range_p
        
        for j in range(1, window):
            curr_norm = (chunk[j] - min_p) / range_p
            dp = curr_norm - prev_norm
            length += np.sqrt(dt_sq + dp**2)
            prev_norm = curr_norm
            
        if length > 0:
            output[i] = 1.0 + (np.log(length) + log_2) / log_2_n_minus_1
        else:
            output[i] = 1.5 # Gaussian Noise default

    return output

@jit(nopython=True, nogil=True, cache=True)
def _numba_rolling_hill(returns_array, window, tail_cut):
    """
    Rolling Hill Estimator for Alpha Stability (Tail Thickness).
    Complexity: O(N * W * log W) due to sorting.
    """
    n = len(returns_array)
    output = np.full(n, np.nan, dtype=np.float64)
    
    # We define min_k to avoid instability
    min_k = 2 
    
    for i in range(window, n):
        chunk = np.abs(returns_array[i-window:i])
        
        # Filter zeros to avoid log(0)
        chunk = chunk[chunk > 1e-9]
        
        len_chunk = len(chunk)
        if len_chunk < 10: # Insufficient data
            continue
            
        # Sort descending (Quick sort)
        # Numba supports np.sort but it sorts ascending
        sorted_chunk = np.sort(chunk)[::-1]
        
        # Determine k (Tail size)
        k = int(len_chunk * tail_cut)
        if k < min_k: 
            k = min_k
            
        # Hill Estimator Logic
        tail_data = sorted_chunk[:k]
        x_min = tail_data[-1] # The threshold
        
        # Avoid division by zero or log of zero
        if x_min <= 1e-9:
            continue
            
        sum_log = 0.0
        for x in tail_data:
            sum_log += np.log(x / x_min)
            
        if sum_log > 0:
            alpha = 1.0 / (sum_log / k)
            output[i] = min(alpha, 3.0) # Cap at 3 (Gaussian is 2, anything > 3 is thin tail)
        else:
            output[i] = 2.0

    return output

# ==========================================
# 2. THE ARCHITECT WRAPPER (Polars Interface)
# ==========================================

class KineticPhaseEnginePolars:
    def __init__(self, fdi_window=30, alpha_window=100, tail_cut=0.05):
        self.fdi_window = fdi_window
        self.alpha_window = alpha_window
        self.tail_cut = tail_cut

    def process_parquet(self, parquet_path, symbol_filter=None):
        """
        Reads Parquet, computes Physics, returns Signal DF.
        """
        print(f"[ARCHITECT] Connecting to {parquet_path}...")
        
        # 1. DuckDB Query (Predicate Pushdown - Load only what is needed)
        con = duckdb.connect()
        query = f"SELECT * FROM '{parquet_path}'"
        if symbol_filter:
            query += f" WHERE symbol = '{symbol_filter}'"
        
        # Zero-Copy convert to Polars LazyFrame
        try:
            lf = pl.from_arrow(con.execute(query).arrow()).lazy()
        except:
             # Fallback for older Polars/DuckDB versions
             lf = pl.scan_parquet(parquet_path)
             if symbol_filter:
                 lf = lf.filter(pl.col("symbol") == symbol_filter)

        
        # 2. Pre-Computation (Lazy)
        # Calculate Returns and Kinetic Components
        lf = lf.sort("time").with_columns([
            pl.col("price").alias("close"), # Rename for consistency if needed, assuming 'price' from binance loader
            pl.col("qty").alias("volume")   # Rename for consistency
        ]).with_columns([
            pl.col("close").pct_change().alias("returns"),
            (pl.col("close") * pl.col("volume")).alias("mass")
        ]).with_columns([
            # Kinetic Energy = 0.5 * Mass * V^2 (Smoothed)
            (0.5 * pl.col("mass") * (pl.col("returns") ** 2))
            .rolling_mean(3).alias("kinetic_energy") # Updated for new Polars syntax
        ])

        # 3. Materialize for Numba (Map Batches)
        # We must collect here because Numba needs contiguous arrays.
        # This is the "Low Memory" compromise: we load columns, not row objects.
        print("[ARCHITECT] Materializing vectors for Physics Engine...")
        df = lf.collect()
        
        if df.height < self.alpha_window:
            print("[WARNING] Not enough data for Alpha Window.")
            return df

        # 4. Apply Numba Kernels
        # We use map_batches to pass the full numpy array to Numba
        
        # A. FDI Calculation
        print("[ARCHITECT] Computing Fractal Dimension (FDI)...")
        close_np = df["close"].to_numpy()
        fdi_values = _numba_rolling_fdi(close_np, self.fdi_window)
        fdi_series = pl.Series("fdi", fdi_values)
        
        # B. Alpha Tail Calculation
        print("[ARCHITECT] Computing Alpha Stability (Hill Estimator)...")
        returns_np = df["returns"].to_numpy()
        alpha_values = _numba_rolling_hill(returns_np, self.alpha_window, self.tail_cut)
        alpha_series = pl.Series("alpha_tail", alpha_values)

        # 5. Attach Results & Generate Signals
        # Note: Polars requires strict re-assignment
        df = df.with_columns([
            fdi_series.alias("fdi"),
            alpha_series.alias("alpha_tail")
        ])
        
        # 6. Logic Gates (Lazy again for speed)
        final_df = df.lazy().with_columns([
            # Gate A: Trend Structure
            (pl.col("fdi") < 1.5).alias("gate_trend"),
            
            # Gate B: Energy Expansion
            (pl.col("kinetic_energy") > pl.col("kinetic_energy").shift(1)).alias("gate_energy"),
            
            # Gate C: Fat Tails (Levy Flight)
            (pl.col("alpha_tail") < 1.7).alias("gate_tails")
        ]).with_columns([
            # ENTRY SIGNAL
            (pl.col("gate_trend") & pl.col("gate_energy") & pl.col("gate_tails"))
            .cast(pl.Int8).alias("signal_raw"),
            
            # EXIT SIGNAL (Entropy Rising or Energy Collapse)
            ((pl.col("kinetic_energy") < pl.col("kinetic_energy").shift(1)) | (pl.col("fdi") > 1.6))
            .cast(pl.Int8).alias("exit_raw")
        ]).with_columns([
            # 7. LAW 3: SHIFT(1) MANDATORY
            pl.col("signal_raw").shift(1).fill_null(0).alias("SIGNAL_ENTRY"),
            pl.col("exit_raw").shift(1).fill_null(0).alias("SIGNAL_EXIT")
        ]).collect()

        return final_df

# ==========================================
# 3. EXECUTION DEMO
# ==========================================
if __name__ == "__main__":
    # Check for real data first
    parquet_file = "data/raw_crypto/ETHUSDT-trades-2023-01.parquet"
    if not os.path.exists(parquet_file):
        print(f"[ARCHITECT] Real data not found at {parquet_file}. Please run binance_loader.py for ETHUSDT.")
        
        # Make dummy
        print("[ARCHITECT] Generating Synthetic Parquet for Audit...")
        dates = pl.datetime_range(start=datetime(2023,1,1), end=datetime(2024,1,1), interval="1h", eager=True)
        # Create a Levy Flight Walk (Random Walk with heavy tails)
        steps = np.random.standard_t(df=1.5, size=len(dates)) # Student-t with df=1.5 has fat tails
        price = 100 * np.exp(np.cumsum(steps * 0.001))
        
        dummy_df = pl.DataFrame({
            "time": dates, # Changed to 'time' to match binance loader
            "symbol": "BTC-USD",
            "price": price, # Changed to 'price'
            "qty": np.random.randint(1000, 10000, len(dates)).astype(float) # Changed to 'qty'
        })
        # Write to temp file
        dummy_df.write_parquet("temp_market_data.parquet")
        parquet_file = "temp_market_data.parquet"

    # Run Engine
    engine = KineticPhaseEnginePolars()
    # No symbol filter needed if file is specific or synthetic has it
    results = engine.process_parquet(parquet_file)
    
    # Tear Sheet Output
    print("\n" + "="*50)
    print("ARCHITECT TEAR SHEET: KINETIC UPGRADE")
    print("="*50)
    trades = results.filter(pl.col("SIGNAL_ENTRY") == 1)
    print(f"Total Bars Processed: {results.height}")
    print(f"Signal Count:         {trades.height}")
    
    if trades.height > 0:
        print("\nSample Signals (Tail):")
        print(trades.select(["time", "close", "fdi", "alpha_tail", "kinetic_energy"]).tail(5))
        
        # Regime Check
        # Use simple indexing
        last_alpha = results["alpha_tail"][-1]
        last_fdi = results["fdi"][-1]
        
        current_regime = "Levy Flight" if last_alpha < 1.7 else "Gaussian Walk"
        print(f"\nCurrent Regime: {current_regime}")
        print(f"Current FDI:    {last_fdi:.4f}")
        print("="*50)
        
    # Cleanup temp
    if parquet_file == "temp_market_data.parquet":
        if os.path.exists("temp_market_data.parquet"):
            os.remove("temp_market_data.parquet")

