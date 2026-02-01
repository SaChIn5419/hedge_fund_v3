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

def run_execution_simulation(results: pl.DataFrame, symbol="ETHUSDT"):
    print(f"\n[ARCHITECT] Initializing Execution Engine for {symbol}...")
    
    # 1. TIME COMPRESSION (Downsampling to reduce noise)
    # We cannot trade every tick. We resample to 1-Second bars for execution logic.
    # This aggregates the burst of 98k signals into manageable trade decisions.
    
    print("[ARCHITECT] Aggregating Nanosecond Physics to 1s Execution Blocks...")
    
    # Cast timestamp to proper datetime for resampling
    # Polars -> Pandas conversion (We accept the RAM hit here for VBT compatibility)
    # Ensure we handle the 'time' column which is Unix ms
    if "time" not in results.columns:
        # Check aliases
        if "timestamp" in results.columns:
            results = results.rename({"timestamp": "time"})
    
    df_exec = results.select([
        (pl.col("time") * 1000).cast(pl.Datetime("us")).alias("datetime"), # time is ms, mult 1000 -> us
        pl.col("close"),
        pl.col("SIGNAL_ENTRY"),
        pl.col("SIGNAL_EXIT")
    ]).to_pandas()
    
    df_exec.set_index("datetime", inplace=True)
    
    # Resample to 5-Second bars (High Freq -> Mid Freq)
    # Logic: If ANY physics signal occurred in the last 5s, we mark 'Signal=True'
    ohlc_dict = {
        'close': 'last',
        'SIGNAL_ENTRY': 'max', # 1 if any signal occurred
        'SIGNAL_EXIT': 'max'
    }
    df_resampled = df_exec.resample("5s").apply(ohlc_dict).dropna()
    
    # Garbage collect the huge tick dataframe
    del df_exec
    gc.collect()
    
    print(f"[ARCHITECT] Execution Blocks Created: {len(df_resampled)} bars")
    
    # 2. DEFINE REALITY (Fees & Slippage)
    # Binance Taker Fee: 0.04% (VIP 0) -> 0.0004
    # Slippage: On 5s aggregation, assume 1 tick slippage (approx 0.01%)
    FEES = 0.0004 
    SLIPPAGE = 0.0001
    
    # 3. RUN SIMULATION
    print("[ARCHITECT] Simulating Orders...")
    
    portfolio = vbt.Portfolio.from_signals(
        close=df_resampled['close'],
        entries=df_resampled['SIGNAL_ENTRY'] == 1,
        exits=df_resampled['SIGNAL_EXIT'] == 1,
        init_cash=100_000,
        fees=FEES,
        slippage=SLIPPAGE,
        size=1.0,         # 100% Equity per trade
        size_type='value',
        freq='5s'         # Frequency for Sharpe calc
    )
    
    return portfolio

def generate_tear_sheet(portfolio):
    print("\n" + "="*50)
    print("ARCHITECT QUANT TEAR SHEET: HFT PHYSICS")
    print("="*50)
    
    # METRICS
    total_return = portfolio.total_return()
    sharpe = portfolio.sharpe_ratio()
    drawdown = portfolio.max_drawdown()
    win_rate = portfolio.stats()['Win Rate [%]']
    trade_count = portfolio.stats()['Total Trades']
    
    print(f"Total Return:    {total_return * 100:.2f}%")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Max Drawdown:    {drawdown * 100:.2f}%")
    print(f"Win Rate:        {win_rate:.2f}%")
    print(f"Trade Count:     {trade_count}")
    print("-" * 50)
    
    # COST ANALYSIS
    total_fees = portfolio.stats()['Total Fees Paid']
    print(f"Fees Paid:       ${total_fees:,.2f}")
    
    if total_fees > 10000:
        print("\n[CRITICAL WARNING] CHURNING DETECTED.")
        print("Strategy is donating alpha to the exchange via fees.")
        print("Recommendation: Increase 'alpha_window' or 'fdi_window'.")
    else:
        print("\n[PASS] Execution Efficiency Acceptable.")
    print("="*50)

# ==========================================
# EXECUTE
# ==========================================
if __name__ == "__main__":
    # 1. Run the Physics Engine First
    parquet_file = "data/raw_crypto/ETHUSDT-trades-2023-01.parquet"
    if not os.path.exists(parquet_file):
        print(f"[ERROR] Data file {parquet_file} not found.")
        # Fallback to dummy generation logic from kinetic engine if needed, 
        # or just rely on kinetic engine to handle missing file gracefully if implemented there
        
    print("[ARCHITECT] Spinning up Kinetic Engine...")
    engine = KineticPhaseEnginePolars()
    results = engine.process_parquet(parquet_file)
    
    # 2. Bridge to VectorBT with Time Compression
    if results is not None and results.height > 0:
        pf = run_execution_simulation(results, symbol="ETHUSDT")
        generate_tear_sheet(pf)
    else:
        print("[ERROR] No results from Kinetic Engine.")
