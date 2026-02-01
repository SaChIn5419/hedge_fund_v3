
import numpy as np
import pandas as pd
import polars as pl
import duckdb
from numba import jit
import sys
import os

# Add parent directory to path to import engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.glass_engine import GlassEngine

# ==========================================
# 1. NUMBA KERNELS (Optimization)
# ==========================================

@jit(nopython=True, nogil=True, cache=True)
def _linregress_slope(x, y):
    """
    Simple linear regression slope calculation: m = (N*sum(xy) - sum(x)*sum(y)) / (N*sum(x^2) - sum(x)^2)
    """
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    
    denom = (n * sum_xx - sum_x * sum_x)
    if denom == 0:
        return 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope

@jit(nopython=True, nogil=True, cache=True)
def _numba_rolling_hurst(log_ret_array, window_size, lags_count=18):
    """
    Calculates Rolling Hurst Exponent via Variance Ratio method.
    lags = range(2, 20) implicitly -> 2..19 (18 lags)
    """
    n = len(log_ret_array)
    output = np.full(n, np.nan, dtype=np.float64)
    
    # Precompute log(lags)
    lags = np.arange(2, 2 + lags_count, dtype=np.float64)
    log_lags = np.log(lags)
    
    # Min required data
    if n < window_size:
        return output
        
    for i in range(window_size, n):
        # Window of log returns
        # We need PRICE series for diff(lag) conceptually, but diff(lag) of price 
        # is roughly sum of log_returns over lag if using log prices.
        # Var(tau) ~ tau^(2H)
        # Var(z(t+tau) - z(t)) where z is log price.
        # z(t+tau) - z(t) = sum(log_ret[t+1 : t+tau])
        
        # Let's reconstruct a local cumulative sum (log price path) for the window
        # to correctly calculate variances of differences.
        window_ret = log_ret_array[i-window_size:i]
        
        # Reconstruct "price" path (relative to 0)
        # method: cumulative sum of log returns
        # This gives log_price profile
        log_price = np.zeros(window_size + 1)
        current_p = 0.0
        log_price[0] = 0.0
        for k in range(window_size):
            current_p += window_ret[k]
            log_price[k+1] = current_p
            
        # efficient variance calculation for each lag
        # var(log_price[t+tau] - log_price[t])
        
        variances = np.zeros(lags_count)
        
        for idx in range(lags_count):
            lag = int(lags[idx])
            # Differences: log_price[t+lag] - log_price[t]
            # Array length: (window_size + 1)
            # Valid indices t: 0 to window_size - lag
            
            diffs_count = (window_size + 1) - lag
            if diffs_count <= 0:
                variances[idx] = np.nan
                continue
                
            # Compute variance of diffs
            # Manual variance to avoid alloc if possible, or just use np.var
            # diffs = log_price[lag:] - log_price[:-lag] (slicing)
            
            # Manual loop for diffs to keep it numba fast/low alloc
            sum_val = 0.0
            sum_sq = 0.0
            
            for t in range(diffs_count):
                val = log_price[t+lag] - log_price[t]
                sum_val += val
                sum_sq += val * val
                
            mean = sum_val / diffs_count
            var = (sum_sq / diffs_count) - (mean * mean)
            
            if var < 1e-9:
                variances[idx] = 1e-9
            else:
                variances[idx] = var
                
        # Fit: log(Var) = 2H * log(tau) + C
        # y = log(variances), x = log_lags
        log_vars = np.log(variances)
        
        slope = _linregress_slope(log_lags, log_vars)
        output[i] = slope / 2.0
        
    return output

@jit(nopython=True, nogil=True, cache=True)
def _numba_rolling_entropy(log_ret_array, window_size, bins=10):
    """
    Calculates Shannon Entropy of return distribution.
    """
    n = len(log_ret_array)
    output = np.full(n, np.nan, dtype=np.float64)
    
    if n < window_size:
        return output

    for i in range(window_size, n):
        window = log_ret_array[i-window_size:i]
        
        # Histogram
        # Numba supports np.histogram
        hist, _ = np.histogram(window, bins)
        
        # Normalize to probability
        # hist is int array
        total = np.sum(hist)
        if total == 0:
            output[i] = 0.0
            continue
            
        entropy = 0.0
        for count in hist:
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
                
        output[i] = entropy
        
    return output

# ==========================================
# 2. STRATEGY PIPELINE
# ==========================================

def run_strategy(parquet_file, symbol="UNKNOWN"):
    print(f"[ARCHITECT] Loading Data for {symbol} from {parquet_file}...")
    
    # 1. Load Data with DuckDB (SQL Speed)
    # We query specific columns and sort by time
    # Adjust for standard yfinance schema (Date, Close)
    query = f"""
        SELECT Date as time, Close as close 
        FROM '{parquet_file}' 
        ORDER BY Date
    """
    df = duckdb.query(query).pl()
    
    # 2. Preprocessing
    print("[ARCHITECT] Preprocessing & Calculating Indicators...")
    
    # Ensure time is Int64 (ms) for GlassEngine
    # If it is datetime, we convert to ms int
    if df["time"].dtype != pl.Int64:
        # Cast to Datetime first to handle Date types, then get epoch ms
        df = df.with_columns([
            pl.col("time").cast(pl.Datetime).dt.epoch("ms").alias("time")
        ])

    # Calculate Log Returns
    # Using Polars for efficient vectorized diff
    # log_ret = log(close / close_shifted) = log(close) - log(close_shifted)
    df = df.with_columns([
        (np.log(pl.col("close") / pl.col("close").shift(1))).alias("log_ret")
    ])
    
    # Convert to Numpy for Numba
    log_ret = df["log_ret"].fill_nan(0.0).to_numpy()
    
    # 3. Physics Calculation (Numba Optimized)
    print("[ARCHITECT] calculating Hurst (Variance Ratio) & Entropy (Shannon)...")
    lookback = 100
    
    hurst = _numba_rolling_hurst(log_ret, lookback)
    entropy = _numba_rolling_entropy(log_ret, lookback)
    
    # 4. Logic Engine (Polars)
    print("[ARCHITECT] Applying Maxwell-Boltzmann Regime Logic...")
    
    # Add metrics back to dataframe
    df_metrics = df.with_columns([
        pl.Series(hurst).fill_nan(0.5).alias("hurts"), # Default 0.5 (Random Walk)
        pl.Series(entropy).fill_nan(0.0).alias("entropy")
    ])
    
    # Rolling Mean for Entropy (Adaptive Threshold)
    # Polars rolling mean
    df_metrics = df_metrics.with_columns([
        pl.col("entropy").rolling_mean(window_size=20).alias("entropy_mean")
    ])
    
    # Define Signals
    # STATE A: Super-Diffusive (Trending) + Low Entropy (Order)
    # STATE B: Mean Reverting
    
    # Conditions
    # Trend Condition: Hurst > 0.6 AND Entropy < Entropy_Mean
    # Reversion Condition: Hurst < 0.4
    
    # For Signal Logic: 
    # Close > Prev Close (Up) or Down for trend following
    # Z-Score < -2.0 for Reversion Buy
    
    # Calculate Z-Score of Close (20 period)
    df_metrics = df_metrics.with_columns([
        pl.col("close").rolling_mean(20).alias("ma_20"),
        pl.col("close").rolling_std(20).alias("std_20")
    ]).with_columns([
        ((pl.col("close") - pl.col("ma_20")) / (pl.col("std_20") + 1e-9)).alias("z_score")
    ])
    
    # Compute Signals in Polars
    # 1 (Buy), -1 (Sell), 0 (Hold)
    
    # We will construct 'entries' and 'exits' boolean arrays for GlassEngine
    # GlassEngine expects: entries (bool), exits (bool)
    # It enters if entries=True and pos=0. Exits if exits=True and pos>0.
    
    # We need to map the user's "Signal = 1/-1" concept to GlassEngine's event loop.
    # User: Signal=1 (Long), Signal=-1 (Short? Or Close Long?)
    # "Sell Trend" usually means Short or Close. 
    # GlassEngine is Long-Only (checks `if position == 0: ... elif position > 0: ...`).
    
    # Assuming Long-Only for now based on GlassEngine implementation.
    # Signal=1 -> Entry.
    # Signal=-1 -> Exit.
    
    cond_trend = (pl.col("hurts") > 0.6) & (pl.col("entropy") < pl.col("entropy_mean"))
    cond_reversion = (pl.col("hurts") < 0.4)
    
    # Trend Buy: Trend Cond & Price Up
    sig_trend_buy = cond_trend & (pl.col("close") > pl.col("close").shift(1))
    
    # Trend Sell (Exit): Trend Cond & Price Down 
    sig_trend_sell = cond_trend & (pl.col("close") < pl.col("close").shift(1))
    
    # Reversion Buy: Reversion Cond & Z-Score < -2 (Oversold)
    sig_reversion_buy = cond_reversion & (pl.col("z_score") < -2.0)
    
    # Combine
    entries_expr = sig_trend_buy | sig_reversion_buy
    exits_expr = sig_trend_sell # Or maybe other exit conditions? 
    # User code had: df.loc[trend_condition & (df['close'] < df['close'].shift(1)), 'Signal'] = -1
    
    # Let's realize these columns
    df_final = df_metrics.with_columns([
        entries_expr.fill_null(False).alias("entry_signal"),
        exits_expr.fill_null(False).alias("exit_signal")
    ])
    
    entries = df_final["entry_signal"].to_numpy()
    exits = df_final["exit_signal"].to_numpy()
    
    # 5. Run Glass Engine
    engine = GlassEngine(initial_capital=100000.0)
    
    # Fix 'time' column for Glass Engine if needed 
    # GlassEngine expects 'time' column to match what run_event_loop receives (int timestamp)
    
    results = engine.run(df_final, entries=entries, exits=exits)
    
    # 6. Report
    print(f"[ARCHITECT] Generating Tearsheet for {symbol}...")
    engine.tear_sheet()

if __name__ == "__main__":
    # Path to a Parquet file
    # Using relative path assuming run from root or handle check
    
    # Candidate: RELIANCE (Liquid)
    fpath = "data/nse/RELIANCE.NS.parquet"
    if not os.path.exists(fpath):
        fpath = os.path.join("architect_v3", fpath)
        
    if os.path.exists(fpath):
        run_strategy(fpath, "RELIANCE")
    else:
        print("File not found. Please check path.")
