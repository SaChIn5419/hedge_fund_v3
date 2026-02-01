import vectorbt as vbt
import polars as pl
import pandas as pd
import numpy as np
import os
import sys
from numba import jit

# Windows Console Encoding Fix
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# INCLUDE NUMBA KERNELS (Required for VBT Factory)
@jit(nopython=True, nogil=True, cache=True)
def _numba_rolling_fdi(price_array, window):
    n = len(price_array)
    output = np.full(n, np.nan, dtype=np.float64)
    log_2 = np.log(2.0)
    log_2_n_minus_1 = np.log(2.0 * (window - 1.0))
    dt_sq = (1.0 / (window - 1.0)) ** 2
    
    for i in range(window, n):
        chunk = price_array[i-window:i]
        max_p = np.max(chunk)
        min_p = np.min(chunk)
        range_p = max_p - min_p
        
        if range_p <= 1e-9: 
            output[i] = 1.0 
            continue
            
        length = 0.0
        prev_norm = (chunk[0] - min_p) / range_p
        
        for j in range(1, window):
            curr_norm = (chunk[j] - min_p) / range_p
            dp = curr_norm - prev_norm
            length += np.sqrt(dt_sq + dp**2)
            prev_norm = curr_norm
            
        if length > 0:
            output[i] = 1.0 + (np.log(length) + log_2) / log_2_n_minus_1
        else:
            output[i] = 1.5 
    return output

@jit(nopython=True, nogil=True, cache=True)
def _numba_rolling_hill(returns_array, window, tail_cut):
    n = len(returns_array)
    output = np.full(n, np.nan, dtype=np.float64)
    min_k = 2 
    
    for i in range(window, n):
        chunk = np.abs(returns_array[i-window:i])
        chunk = chunk[chunk > 1e-9]
        len_chunk = len(chunk)
        if len_chunk < 10: 
            continue
        sorted_chunk = np.sort(chunk)[::-1]
        k = int(len_chunk * tail_cut)
        if k < min_k: k = min_k
        tail_data = sorted_chunk[:k]
        x_min = tail_data[-1] 
        if x_min <= 1e-9: continue
        sum_log = 0.0
        for x in tail_data:
            sum_log += np.log(x / x_min)
        if sum_log > 0:
            alpha = 1.0 / (sum_log / k)
            output[i] = min(alpha, 3.0) 
        else:
            output[i] = 2.0
    return output

def run_hyper_optimization(parquet_path="data/raw_crypto/ETHUSDT-trades-2023-01.parquet"):
    print("[ARCHITECT] Initiating Hyper-Grid Optimization Protocol...")
    
    if not os.path.exists(parquet_path):
        print(f"[ERROR] Parquet file not found at {parquet_path}")
        return

    # 1. LOAD DATA (Polars)
    # Use lazy scan for initial efficiency, but read into memory for processing
    print(f"[ARCHITECT] Loading {parquet_path}...")
    df = pl.read_parquet(parquet_path).sort("time").with_columns([
        pl.col("price").alias("close")
    ])
    
    # 2. DOWNSAMPLE FOR OPTIMIZATION SPEED
    # Optimization on tick data is too slow. We optimize on 1-minute bars.
    print(f"[ARCHITECT] Aggregating {len(df)} ticks to 1-Minute OHLCV...")
    
    # Use pandas for resampling convenience (as requested in prompting)
    # Convert 'time' (unix ms) to datetime
    df_pd = df.select([
        (pl.col("time") * 1000).cast(pl.Datetime("us")).alias("datetime"),
        pl.col("close").alias("price") # Ensure 'close' exists or rename 'price'
    ]).to_pandas().set_index("datetime")
    
    # Resample to 1 minute
    price_1m_series = df_pd["price"].resample("1min").last().dropna()
    close_1m = price_1m_series.to_numpy()
    
    print(f"[ARCHITECT] Optimized Vector Length: {len(close_1m)}")

    # 3. DEFINE INDICATOR FACTORY
    # We wrap our Numba logic into a VBT Indicator for broadcasting
    
    def apply_physics(close, fdi_w, alpha_w, tail_cut):
        # Calculate Returns inside the factory
        # Note: close is a 2D array (broadcasting) if params are broadcasted, but here VBT handles it
        # VBT apply func typically receives 1D array per column if vectorized=False or 2D if vectorized=True
        # For simple custom funcs with Numba, it's best to rely on VBT's apply_on_axis logic implicitly
        # But wait, numba functions expect 1D arrays. 
        # VBT IndicatorFactory with 'param_names' will run this function for each connection of parameters.
        # It's safest to treat 'close' as 1D here and let VBT handle the loop over params.
        
        # Returns
        # Robust Returns Calc
        # Force 1D for safety if VBT passes 1D
        close_1d = np.asarray(close).flatten()
        ret = np.concatenate((np.array([0.0]), np.diff(close_1d))) / (close_1d + 1e-9)
        
        fdi = _numba_rolling_fdi(close_1d, fdi_w)
        alpha = _numba_rolling_hill(ret, alpha_w, tail_cut)
        
        return fdi, alpha

    PhysicsInd = vbt.IndicatorFactory(
        class_name="Physics",
        short_name="phys",
        input_names=["close"],
        param_names=["fdi_w", "alpha_w", "tail_cut"],
        output_names=["fdi", "alpha"]
    ).from_apply_func(apply_physics)

    # 4. RUN INDICATOR OVER PARAMETER GRID
    # Ranges
    fdi_windows = [30, 60, 120]
    alpha_windows = [100, 200, 300]
    tail_cuts = [0.05] # Keep constant for now
    
    print("[ARCHITECT] Computing Physics Grid...")
    # This runs the heavy math for ALL combinations at once via broadcasting
    res = PhysicsInd.run(
        price_1m_series, # Pass the Series so VBT knows the index
        fdi_w=fdi_windows, 
        alpha_w=alpha_windows, 
        tail_cut=tail_cuts,
        param_product=True
    )

    # 5. SIMULATE SIGNALS (Iterative Grid for Thresholds)
    # VBT Param broadcasting on top of Indicator broadcasting can be complex.
    # We will loop over thresholds and leverage the already-vectorized indicator results.
    
    alpha_thres_list = [1.5, 1.6, 1.7]
    fdi_entry_list = [1.5]
    fdi_exit_list = [1.6]
    
    print("[ARCHITECT] Simulating Execution Scenarios (Threshold Grid)...")
    
    results_list = []
    
    import itertools
    for a_th, f_en, f_ex in itertools.product(alpha_thres_list, fdi_entry_list, fdi_exit_list):
        # res.alpha and res.fdi are DataFrames (Time x WindowParams)
        # Broadcasting scalar threshold across the DataFrame
        entries = (res.alpha < a_th) & (res.fdi < f_en)
        exits = res.fdi > f_ex
        
        # Run Portfolio for this threshold combo (vectorized across all window combos)
        # We need to ensure columns match. entries/exits have same columns as close_1m?
        # No, close_1m is 1D array. VBT broadcasts close to match entries shape.
        pf = vbt.Portfolio.from_signals(
            close=price_1m_series, # Use Series for index alignment
            entries=entries,
            exits=exits,
            fees=0.0004,
            slippage=0.0001,
            freq='1min',
            init_cash=100000
        )
        
        # Aggregating Results
        # pf.total_return() is a Series with index = Window Params
        total_ret = pf.total_return()
        sharpe = pf.sharpe_ratio()
        trades = pf.trades.count()
        
        # Create temporary DF for this threshold set
        tmp = pd.DataFrame({
            "Sharpe": sharpe,
            "Return": total_ret,
            "Trades": trades
        })
        # Add threshold info
        tmp["Alpha_Th"] = a_th
        tmp["FDI_Entry"] = f_en
        tmp["FDI_Exit"] = f_ex
        
        results_list.append(tmp)

    # 6. RANKING & TEAR SHEET
    print("\n[ARCHITECT] OPTIMIZATION COMPLETE. TOP CONFIGURATIONS:")
    print("="*60)
    
    if not results_list:
        print("[WARNING] No results generated.")
        return None

    # Concatenate all results
    metrics = pd.concat(results_list)
    
    # Filter for sane trade counts (e.g., > 10 but < 500)
    viable = metrics[(metrics["Trades"] > 10) & (metrics["Trades"] < 5000)]
    
    # Sort by Sharpe
    top_results = viable.sort_values("Sharpe", ascending=False).head(5)
    
    print(top_results)
    print("="*60)
    
    if not top_results.empty:
        # Re-construct the winner is tricky with the loops. 
        # Just printing the params is enough for the user to act.
        best_cfg = top_results.iloc[0]
        print(f"\n[WINNER DETAILS]\n{best_cfg}")
        return None # Return logic simplified for now
    else:
        print("[WARNING] No viable configurations found.")
        return None

if __name__ == "__main__":
    run_hyper_optimization()
