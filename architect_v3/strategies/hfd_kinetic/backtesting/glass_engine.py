import polars as pl
import numpy as np
import pandas as pd
from numba import jit
import warnings
import sys
import os

# Ensure import of VizEngine
try:
    from strategies.hfd_kinetic.backtesting.viz_engine import VizEngine
except ImportError:
    try:
        from .viz_engine import VizEngine
    except ImportError:
        from viz_engine import VizEngine

# ==========================================
# 1. PHYSICS KERNELS (The Math)
# ==========================================
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
            output[i] = 1.0; continue
            
        length = 0.0
        prev_norm = (chunk[0] - min_p) / range_p
        for j in range(1, window):
            curr_norm = (chunk[j] - min_p) / range_p
            dp = curr_norm - prev_norm
            length += np.sqrt(dt_sq + dp**2)
            prev_norm = curr_norm
            
        if length > 0: output[i] = 1.0 + (np.log(length) + log_2) / log_2_n_minus_1
        else: output[i] = 1.5
    return output

@jit(nopython=True, nogil=True, cache=True)
def _numba_rolling_hill(returns_array, window, tail_cut):
    n = len(returns_array)
    output = np.full(n, np.nan, dtype=np.float64)
    min_k = 2
    
    for i in range(window, n):
        chunk = np.abs(returns_array[i-window:i])
        chunk = chunk[chunk > 1e-9]
        if len(chunk) < 10: continue
            
        sorted_chunk = np.sort(chunk)[::-1]
        k = max(int(len(chunk) * tail_cut), min_k)
        
        tail_data = sorted_chunk[:k]
        x_min = tail_data[-1]
        if x_min <= 1e-9: continue
            
        sum_log = 0.0
        for x in tail_data: sum_log += np.log(x / x_min)
        
        output[i] = min(1.0 / (sum_log / k), 3.0) if sum_log > 0 else 2.0
    return output

# ==========================================
# 2. THE GLASS ENGINE (Event Loop)
# ==========================================
@jit(nopython=True, nogil=True, cache=True)
def run_event_loop(prices, entries, exits, leverage_array, timestamps, init_cash, fee_pct, slippage):
    """
    THE CORE ENGINE: Compiles to Machine Code via Numba.
    Iterates through time, managing account state like a real exchange.
    Supports Dynamic Leverage/Position Sizing.
    """
    n = len(prices)
    
    # State Variables
    cash = float(init_cash)
    position = 0.0
    equity_curve = np.zeros(n)
    equity_curve[0] = init_cash
    
    # Trade Log Arrays (Max size = n)
    trade_entry_idx = np.zeros(n, dtype=np.int64)
    trade_exit_idx = np.zeros(n, dtype=np.int64)
    trade_pnl = np.zeros(n)
    trade_count = 0
    
    for i in range(1, n):
        # 1. Update Equity (Mark to Market)
        current_price = prices[i]
        equity = cash + (position * current_price)
        equity_curve[i] = equity
        
        # 2. Check Signals
        # We assume Instant Execution at Close for simplicity
        
        signal_buy = entries[i]
        signal_sell = exits[i]
        
        # LOGIC: LONG ONLY REGIME
        if position == 0:
            if signal_buy:
                # ENTRY EXECUTION
                # Apply Slippage
                exec_price = current_price * (1 + slippage)
                
                # Dynamic Leverage
                lev = leverage_array[i]
                if lev <= 0.0: continue # Skip if 0 leverage
                
                # Calculate Size (Equity * Leverage)
                # Ensure we don't spend more than we have for cash check? 
                # GlassEngine is cash-based. If leverage > 1, we imply margin.
                # Use simplified "Cash * Leverage" or "Equity * Leverage"?
                # Standard: Target Exposure = Equity * Leverage.
                target_exposure = equity * lev
                
                cost = target_exposure
                # Check if we have enough cash? For >1 leverage we assume margin availability.
                # For <1 leverage we just use less cash.
                # Let's subtract cost from cash. If cash goes negative, we are on margin.
                # GlassEngine v1.0 Model: Pure Cash (no margin interest implemented yet).
                
                if cost > 0:
                    size = cost / exec_price
                    
                    # Fee Deduction (on Notional)
                    fee = (size * exec_price) * fee_pct
                    
                    # Update Cash: Cash decreases by purchase cost + fee
                    cash -= (size * exec_price) + fee
                    position = size
                    
                    # Log Trade Entry
                    trade_entry_idx[trade_count] = i
                
        elif position > 0:
            if signal_sell:
                # EXIT EXECUTION
                exec_price = current_price * (1 - slippage)
                
                # Fee Deduction
                value = position * exec_price
                fee = value * fee_pct
                
                cash += value - fee
                
                # Log Trade Exit
                trade_exit_idx[trade_count] = i
                
                # Calculate PnL
                entry_price_ref = prices[trade_entry_idx[trade_count]]
                trade_pnl[trade_count] = (exec_price - entry_price_ref) / entry_price_ref
                
                position = 0.0
                trade_count += 1

    # Truncate logs
    return equity_curve, trade_entry_idx[:trade_count], trade_exit_idx[:trade_count], trade_pnl[:trade_count]

class GlassEngine:
    def __init__(self, fee=0.0004, slippage=0.0001, initial_capital=100000.0):
        self.fee = fee
        self.slippage = slippage
        self.initial_capital = initial_capital
        
    def run(self, df: pl.DataFrame):
        """
        Orchestrates the simulation using Polars Data.
        """
        print("[ARCHITECT] Spinning up Glass Engine...")
        
        # 1. Prepare Data
        df = df.sort("time")
        times = df["time"].to_numpy() # Unix MS
        closes = df["close"].to_numpy()
        alpha = df["alpha_tail"].to_numpy()
        fdi = df["fdi"].to_numpy()
        
        # Golden Params
        entries = (alpha < 1.6) & (fdi < 1.5)
        exits = (fdi > 1.6)
        
        # 2. Run Simulation
        print("[ARCHITECT] Executing Trade Loop...")
        equity, entry_idx, exit_idx, pnl = run_event_loop(
            closes, entries, exits, times, 
            self.initial_capital, self.fee, self.slippage
        )
        
        # 3. Compile Results
        self.results = {
            "equity_curve": equity,
            "trades": pd.DataFrame({
                "entry_time": times[entry_idx],
                "exit_time": times[exit_idx],
                "entry_price": closes[entry_idx],
                "exit_price": closes[exit_idx],
                "pnl_pct": pnl
            })
        }
        
        return self.results

    def tear_sheet(self):
        trades = self.results["trades"]
        equity = self.results["equity_curve"]
        
        from engine.viz_engine import VizEngine # Local import to ensure availability
        
        # We need the Datetime Index for the equity curve
        # We have 'times' array in self.results["equity_curve"]? No, we didn't save times there.
        # But we passed 'times' to run.
        # Wait, self.results keys: 'equity_curve', 'trades'.
        # We need to reconstruction the time index.
        # It was passed into `run_event_loop` as `times` (Unix ms).
        # We should store it in self.results.
        
        # Ah, I don't have the time array in self.results in the previous implementation.
        # Let's check `run` method.
        pass
        
    def run(self, df: pl.DataFrame, entries=None, exits=None, leverage=None):
        """
        Orchestrates the simulation using Polars Data.
        """
        print("[ARCHITECT] Spinning up Glass Engine...")
        
        # 1. Prepare Data
        df = df.sort("time")
        times = df["time"].to_numpy() # Unix MS
        closes = df["close"].to_numpy()
        
        # Golden Params (Default if no signals provided)
        if entries is None or exits is None:
            alpha = df["alpha_tail"].to_numpy()
            fdi = df["fdi"].to_numpy()
            entries = (alpha < 1.6) & (fdi < 1.5)
            exits = (fdi > 1.6)

        # Default Leverage (1.0) if not provided
        if leverage is None:
            leverage = np.ones(len(closes), dtype=np.float64)
        
        # 2. Run Simulation
        print("[ARCHITECT] Executing Trade Loop...")
        equity, entry_idx, exit_idx, pnl = run_event_loop(
            closes, entries, exits, leverage, times, 
            self.initial_capital, self.fee, self.slippage
        )
        
        # 3. Compile Results
        self.results = {
            "equity_curve": equity,
            "times": times, # ADDED THIS FOR VIZ ENGINE
            "trades": pd.DataFrame({
                "entry_time": times[entry_idx],
                "exit_time": times[exit_idx],
                "entry_price": closes[entry_idx],
                "exit_price": closes[exit_idx],
                "pnl_pct": pnl
            })
        }
        
        return self.results

    def tear_sheet(self):
        if self.results is None: return
        
        # Create Time Index
        time_idx = pd.to_datetime(self.results["times"], unit='ms')
        
        # Ignite Viz Engine
        viz = VizEngine(self.results, time_idx)
        viz.report()
        
        return self.results["trades"]

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    parquet_path = "data/raw_crypto/ETHUSDT-trades-2023-01.parquet"
    if not os.path.exists(parquet_path):
        print(f"[ERROR] File not found: {parquet_path}")
        sys.exit(1)

    print("[ARCHITECT] Loading Data for Glass Engine...")
    df = pl.read_parquet(parquet_path).sort("time").with_columns([
        pl.col("price").alias("close")
    ])

    print("[ARCHITECT] Aggregating to 1-Minute Bars...")
    # Preserve explicit 'time' as int64 ms for the Glass Engine
    df_1m = df.with_columns([
        pl.col("time").cast(pl.Datetime("ms")).alias("datetime")
    ]).group_by_dynamic("datetime", every="1m").agg([
        pl.col("close").last(),
        pl.col("time").last().alias("time") # Keep underlying int ms
    ]).sort("datetime")

    print("[ARCHITECT] Computing Physics Inputs (FDI 30, Alpha 100)...")
    close_np = df_1m["close"].to_numpy()
    # Robust returns calc
    returns_np = np.concatenate((np.array([0.0]), np.diff(close_np))) / (close_np + 1e-9)
    
    # Calculate Physics
    fdi = _numba_rolling_fdi(close_np, 30)
    alpha = _numba_rolling_hill(returns_np, 100, 0.05)
    
    # Add back to Polars
    results = df_1m.with_columns([
        pl.Series(fdi).alias("fdi"),
        pl.Series(alpha).alias("alpha_tail")
    ])

    # Run Glass Engine
    bt = GlassEngine(fee=0.0004, slippage=0.0001)
    results_dict = bt.run(results)
    bt.tear_sheet()
