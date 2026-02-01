import vectorbt as vbt
import polars as pl
import pandas as pd
import numpy as np
import gc
import sys
import os
from kinetic_engine_numba import KineticPhaseEnginePolars

# Windows Console Encoding Fix
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def run_regime_strategy(results: pl.DataFrame, symbol="ETHUSDT"):
    print(f"\n[ARCHITECT] Initializing V3: Regime Swing Engine for {symbol}...")
    
    # 1. INTELLIGENT AGGREGATION (Ticks -> 1 Min)
    print("[ARCHITECT] Downsampling Physics to 1-Minute OHLCV...")
    
    # Cast timestamp to proper datetime for resampling
    # Ensure 'time' column is handled correctly (Unix ms from Binance)
    # Check column name again
    cols = results.columns
    time_col = "time" if "time" in cols else "timestamp"
    
    # Convert to Pandas for Resampling (Ram intensive but necessary for reliable indexing)
    # Using 'us' precision for Datetime
    df_raw = results.select([
        (pl.col(time_col) * 1000).cast(pl.Datetime("us")).alias("datetime"),
        pl.col("close"),
        pl.col("fdi"),
        pl.col("alpha_tail"),
        pl.col("kinetic_energy")
    ]).to_pandas().set_index("datetime")
    
    # Resample Logic:
    # Price -> Last
    # FDI/Alpha -> Min (We want to capture the *most extreme* physics reading in that minute)
    # Energy -> Sum
    resampler = df_raw.resample("1min")
    df_1m = pd.DataFrame()
    df_1m['close'] = resampler['close'].last()
    df_1m['fdi_min'] = resampler['fdi'].min()       # Did the market trend ANY time during this minute?
    df_1m['alpha_min'] = resampler['alpha_tail'].min() # Did a tail event occur?
    df_1m['energy_sum'] = resampler['kinetic_energy'].sum()
    
    # Drop empty bins
    df_1m.dropna(inplace=True)
    
    print(f"[ARCHITECT] Market Regime Blocks: {len(df_1m)}")
    
    # 2. SIGNAL LOGIC (The Filter)
    # ENTRY: Extreme Physics detected within the minute
    entries = (df_1m['alpha_min'] < 1.7) & (df_1m['fdi_min'] < 1.4)
    
    # EXIT: Regime Shift (Market becomes random noise)
    # We smooth the exit condition to avoid "flickering"
    # Exit if FDI stays high (random) for 5 minutes
    fdi_ma = df_1m['fdi_min'].rolling(5).mean()
    exits = fdi_ma > 1.6
    
    # 3. EXECUTION SIMULATION (Fixed Sizing)
    FEES = 0.0004     # 0.04% Taker
    SLIPPAGE = 0.0001 # 1 basis point (optimistic on 1m candles)
    
    print("[ARCHITECT] Engaging VectorBT Engine...")
    portfolio = vbt.Portfolio.from_signals(
        close=df_1m['close'],
        entries=entries,
        exits=exits,
        init_cash=100000,
        fees=FEES,
        slippage=SLIPPAGE,
        size=1.0,             # 100% of...
        size_type='percent',  # ...Current Equity (Compounding enabled)
        freq='1min'
    )
    
    return portfolio

def generate_v3_tear_sheet(portfolio):
    print("\n" + "="*50)
    print("ARCHITECT QUANT TEAR SHEET: REGIME SWING (V3)")
    print("="*50)
    
    # Use default stats to avoid version-specific KeyErrors
    stats = portfolio.stats()
    
    print(stats.to_string())
    print("-" * 50)
    
    # Check if we are actually holding
    avg_dur = portfolio.trades.duration.mean()
    print(f"Avg Hold Time:   {avg_dur}")
    
    # Fees check
    total_fees = portfolio.stats()['Total Fees Paid']
    print(f"Fees Paid:       ${total_fees:,.2f}")
    print("="*50)

# ==========================================
# EXECUTE V3
# ==========================================
if __name__ == "__main__":
    # 1. Run the Physics Engine First
    parquet_file = "data/raw_crypto/ETHUSDT-trades-2023-01.parquet"
    if not os.path.exists(parquet_file):
        print(f"[ERROR] Data file {parquet_file} not found.")
    else:
        print("[ARCHITECT] Spinning up Kinetic Engine (Numba)...")
        engine = KineticPhaseEnginePolars()
        results = engine.process_parquet(parquet_file)
        
        # 2. Run Regime Swing Strategy
        if results is not None and results.height > 0:
            pf_v3 = run_regime_strategy(results, symbol="ETHUSDT")
            generate_v3_tear_sheet(pf_v3)
        else:
            print("[ERROR] No results from Kinetic Engine.")
