
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

def run_kinetic_strategy(parquet_file, symbol="UNKNOWN"):
    print(f"[ARCHITECT] Loading Data for {symbol} from {parquet_file}...")
    
    # 1. Load Data with DuckDB
    query = f"""
        SELECT Date as time, 
               Close as close, 
               High as high, 
               Low as low 
        FROM '{parquet_file}' 
        ORDER BY Date
    """
    df = duckdb.query(query).pl()
    
    # Ensure time is ms (if ns)
    if df["time"].dtype != pl.Int64:
        df = df.with_columns([
            pl.col("time").cast(pl.Datetime).dt.epoch("ms").alias("time")
        ])
    
    print("[ARCHITECT] Calculating Kinetic Metrics (Polars Optimized)...")
    
    # 2. Physics Filter: Volatility Regime (ATR / Close)
    # ATR = Rolling Mean of True Range
    # TR = Max(High-Low, Abs(High-PreClose), Abs(Low-PreClose))
    
    # Calculate PreClose
    df = df.with_columns([
        pl.col("close").shift(1).alias("prev_close")
    ])
    
    # Calculate TR Components
    # We use Polars expressions for speed
    df = df.with_columns([
        (pl.col("high") - pl.col("low")).alias("tr1"),
        (pl.col("high") - pl.col("prev_close")).abs().alias("tr2"),
        (pl.col("low") - pl.col("prev_close")).abs().alias("tr3")
    ])
    
    # TR is Max of tr1, tr2, tr3
    df = df.with_columns([
        pl.max_horizontal(["tr1", "tr2", "tr3"]).alias("tr")
    ])
    
    # ATR (14) -> Wilder's Smoothing usually, but user code used 'ta.atr'.
    # ta.atr defaults to RMA (Wilders). Polars ewm_mean(alpha=1/14, adjust=False) approximates RMA?
    # Actually alpha=1/length is effectively RMA for large N.
    # Let's use simple rolling mean or ewm for speed. User code `ta.atr` uses RMA.
    # We will use ewm_mean with com=(length-1) which is standard for EMA, 
    # but RMA is alpha=1/L. alpha = 1 / (1 + com). So 1/14 = 1/(1+com) -> 1+com = 14 -> com=13.
    # Polars ewm_mean uses 'com' or 'span'.
    # Span=2N-1? No.
    # Let's stick to ewm_mean(alpha=1/14).
    
    # Filter Re-calculation
    df = df.with_columns([
        pl.col("tr").ewm_mean(alpha=1.0/14.0, adjust=False, min_periods=14).alias("atr")
    ])

    # Vol Regime = ATR / Close
    df = df.with_columns([
        (pl.col("atr") / pl.col("close")).alias("vol_regime")
    ])

    # 3. Engine A: Short-Term Momentum (Velocity)
    # Fast EMA (9) vs Slow EMA (21)
    df = df.with_columns([
        pl.col("close").ewm_mean(span=9, adjust=False).alias("fast_ma"),
        pl.col("close").ewm_mean(span=21, adjust=False).alias("slow_ma")
    ])

    # 4. Volatility Target (Hallucination Guardrail)
    # Target Vol = 1.5% daily
    # Leverage = Target / Vol_Regime
    target_vol = 0.015
    df = df.with_columns([
        (target_vol / pl.col("vol_regime")).clip(0.0, 1.0).alias("leverage_calc")
    ])

    # 5. Signal Generation
    # Long: Fast > Slow AND Vol_Regime < 0.05
    # Exit: Fast < Slow
    
    cond_long = (pl.col("fast_ma") > pl.col("slow_ma")) & (pl.col("vol_regime") < 0.05)
    cond_exit = (pl.col("fast_ma") < pl.col("slow_ma"))

    # We need to shift Signal logic?
    # User code:
    # df.loc[long_condition, 'Signal'] = 1
    # df.loc[exit_condition, 'Signal'] = 0
    # Then df['Final_Signal'] = df['Signal'].shift(1)
    # Then df['Position_Size'] = df['Final_Signal'] * df['leverage'].shift(1)
    
    # In GlassEngine, we calculate 'entries' (boolean) and 'exits' (boolean).
    # GlassEngine enters if entries[i] is True.
    # If the user wants to enter at Time T based on Signal computed at T (using T close),
    # in reality we enter at T+1 Open.
    # GlassEngine simulates "Instant Execution at Close" currently (Line 100 in engine).
    # So if we want to match reality, we should shift signals by 1.
    # User code explicitly shifts: `df['Final_Signal'] = df['Signal'].shift(1)`.
    # So we should shift our boolean conditions.
    
    # Logic in Polars:
    # 1. Create raw Signal column (1 or 0)
    # 2. Shift Signal
    # 3. Create Entries/Exits from Shifted Signal
    
    df = df.with_columns([
        pl.when(cond_long).then(1)
          .when(cond_exit).then(0)
          .otherwise(pl.col("close") * 0) # Default 0? Or Forward Fill?
          # User code: df['Signal'] = 0 initially. Then sets 1 where long, 0 where exit.
          # If neither condition met? It stays 0 (because initialized to 0).
          # Wait, if Fast > Slow and Vol > 0.05? It's not Long. It's 0.
          # So 0 is correct default.
          .alias("signal_raw")
    ])
    
    # The user logic implies:
    # If currently holding (Signal=1), and next raw is 0, we exit.
    # If currently flat (Signal=0), and next raw is 1, we enter.
    
    # Shift Signal to be 'Final_Signal'
    df = df.with_columns([
        pl.col("signal_raw").shift(1).fill_null(0).alias("final_signal"),
        pl.col("leverage_calc").shift(1).fill_null(1.0).alias("final_leverage")
    ])
    
    # Construct Boolean Triggers for GlassEngine
    # Entry: Final_Signal == 1 (and we are not in position - handled by Engine)
    # Exit: Final_Signal == 0 (and we are in position - handled by Engine)
    
    entries = (df["final_signal"] == 1).to_numpy()
    exits = (df["final_signal"] == 0).to_numpy()
    leverage = df["final_leverage"].to_numpy()
    
    # 6. Run Glass Engine
    engine = GlassEngine(initial_capital=100000.0)
    results = engine.run(df, entries=entries, exits=exits, leverage=leverage)
    
    # 7. Report
    print(f"[ARCHITECT] Generating Tearsheet for {symbol}...")
    engine.tear_sheet()

if __name__ == "__main__":
    # Candidate: RELIANCE
    fpath = "data/nse/RELIANCE.NS.parquet"
    if not os.path.exists(fpath):
        fpath = os.path.join("architect_v3", fpath)
        
    if os.path.exists(fpath):
        run_kinetic_strategy(fpath, "RELIANCE")
    else:
        print("File not found.")
