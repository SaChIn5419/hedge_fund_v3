import polars as pl
import numpy as np
import pandas as pd
import warnings
import sys
import os

# Import Glass Engine and Kernels from the existing file
# Ensure we are in the correct directory for imports
sys.path.append(os.getcwd())
try:
    from engine.glass_engine import GlassEngine, _numba_rolling_fdi, _numba_rolling_hill
except ImportError:
    # Fallback if running from within engine dir
    from glass_engine import GlassEngine, _numba_rolling_fdi, _numba_rolling_hill

# Windows Console Encoding Fix
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

warnings.filterwarnings('ignore')

def generate_oos_data():
    """
    Generates 'Out of Sample' data.
    Regime: High Volatility Chop (The 'Widowmaker' Regime).
    """
    print("[ARCHITECT] Generating OOS Stress Test Data (Regime: CHOP)...")
    # Generate 1 month of high-frequency data (1s freq)
    # 30 days * 24 * 60 * 60 = 2,592,000 points
    # This might be too heavy for O-U loop in pure python?
    # Let's do 5 days to be safe and fast? Or just let it run.
    # User code suggested 2023-02-01 to 2023-03-01. That's 2.4M points.
    # Python loop might be slow. I'll vectorize the O-U process.
    
    dates = pd.date_range(start="2023-02-01", end="2023-03-01", freq="1s")
    n = len(dates)
    
    # Vectorized O-U Process Approximation
    # dx = theta*(mu-x)*dt + sigma*dW
    # We can cumsum deviations? No, mean reversion depends on current level.
    # Fast iteration or Numba? We'll use a simple loop, it's 2M iterations, takes a few seconds in Python.
    
    prices = np.zeros(n)
    
    # Initial Price
    current = 1600.0
    mu = 1600.0
    theta = 0.1
    sigma = 2.0
    dt = 1.0 # 1 second steps
    
    # Precompute noise
    noise = np.random.normal(0, 1, n)
    
    # Use simple Numba loop for generation speed if available, or python loop
    # We can define a helper for speed
    prices = fast_ou_process(n, current, mu, theta, sigma, noise)
        
    # Inject Levy Flights (Jumps)
    print("[ARCHITECT] Injecting Levy Flight Shocks...")
    jumps = np.random.choice(range(n-1000), size=50, replace=False)
    for j in jumps:
        # Shock magnitude
        shock = np.random.normal(0, 20) 
        # Sustained shock? Or just a blip? user code: prices[j:j+1000] += ...
        # This implies a regime shift (step change) that lasts 1000 seconds
        prices[j:j+1000] += shock
        
    df = pl.DataFrame({
        "time": dates.astype(np.int64) // 10**6, # ms timestamp
        "close": prices,
        "symbol": "ETH-USD"
    })
    
    return df

from numba import jit

@jit(nopython=True)
def fast_ou_process(n, start, mu, theta, sigma, noise):
    prices = np.zeros(n)
    prices[0] = start
    dt = 1.0/n # The user code had (1/n) in the dx term: theta * (mu - prev) * (1/n)
    # The snippet: dx = theta * (mu - prices[t-1]) * (1/n) + sigma * np.random.normal()
    # 1/n implies the time step matches the array length relative to unit time? 
    # Usually dt is time delta. If freq="1s" over a month, n is large. 
    # 1/2.6M is tiny dt. 
    # Let's stick strictly to user provided logic: `(1/n)`
    inv_n = 1.0 / n
    x = start
    for t in range(1, n):
        dx = theta * (mu - x) * inv_n + sigma * noise[t]
        x = x + dx
        prices[t] = x
    return prices

def run_oos_test():
    # 1. Generate Data
    df_oos = generate_oos_data()
    
    # 2. Compute Physics (Strictly adherence to Golden Config)
    print("[ARCHITECT] Computing OOS Physics...")
    
    # Aggregate to 1m
    # Use appropriate cast for ms timestamps
    df_1m = df_oos.with_columns([
        (pl.col("time") * 1000).cast(pl.Datetime("us")).alias("datetime")
    ]).group_by_dynamic("datetime", every="1m").agg([
        pl.col("close").last(),
        pl.col("time").last() # Keep unix time for engine
    ]).sort("datetime")
    
    # Numba Calculation
    close_vals = df_1m["close"].to_numpy()
    
    # Robust Returns
    ret_vals = np.concatenate((np.array([0.0]), np.diff(close_vals))) / (close_vals + 1e-9)
    
    # GOLDEN PARAMS
    fdi = _numba_rolling_fdi(close_vals, 30)
    alpha = _numba_rolling_hill(ret_vals, 100, 0.05)
    
    # 3. Prepare Dataframe for Glass Engine
    df_sim = pl.DataFrame({
        "time": df_1m["time"],
        "close": df_1m["close"],
        "fdi": fdi,
        "alpha_tail": alpha
    })
    
    # 4. Execute Glass Engine
    engine = GlassEngine(fee=0.0004, slippage=0.0001, initial_capital=100000.0)
    results = engine.run(df_sim)
    
    # 5. Fix Reporting Bug & Report
    # Ensure equity curve starts at initial capital visually
    results["equity_curve"][0] = 100000.0 
    
    print("\n[ARCHITECT] RUNNING OOS FORENSICS...")
    trades = engine.tear_sheet()
    
    return trades

if __name__ == "__main__":
    run_oos_test()
