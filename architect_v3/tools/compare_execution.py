import yaml
import sys
import os
import polars as pl
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.alpha_v1 import DualEngineAlpha
from engine.core_physics import VectorizedCore

# Terminal colour constants
PRIME = "\033[92m"
RESET = "\033[0m"

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def calculate_metrics(df, name="Strategy"):
    initial = df["equity"][0]
    final = df["equity"][-1]
    total_ret = (final / initial) - 1
    
    # CAGR
    days = df.height
    years = days / 252
    cagr = (final / initial) ** (1/years) - 1
    
    # Sharpe
    ret = df["equity"].pct_change().drop_nulls()
    sharpe = (ret.mean() * 252 - 0.06) / (ret.std() * np.sqrt(252))
    
    # Max DD
    peak = df["equity"].cum_max()
    dd = (df["equity"] / peak) - 1
    max_dd = dd.min()
    
    return {
        "Name": name,
        "Total Return": f"{total_ret*100:.2f}%",
        "CAGR": f"{cagr*100:.2f}%",
        "Sharpe": f"{sharpe:.2f}",
        "Max DD": f"{max_dd*100:.2f}%",
        "Final Equity": f"{final:,.0f}"
    }

def run():
    print(f"{PRIME}ARCHITECT V3: EXECUTION DELAY COMPARISON{RESET}")
    print("-------------------------------------------------")
    cfg = load_config()

    alpha = DualEngineAlpha(momentum_window=cfg['alpha_model']['momentum_window'])
    core = VectorizedCore(cfg['universe']['nse_parquet_path'], alpha)
    
    # RUN 1: T+0 (Standard)
    print("\n>>> RUNNING SIMULATION 1: T+0 Execution (Close-to-Close)")
    eq_t0 = core.run_historical_backtest(
        initial_capital=1000000.0,
        top_n=cfg['alpha_model']['max_positions'],
        energy_threshold=cfg['alpha_model']['energy_threshold'],
        start_date="2015-01-01",
        execution_delay=0  # NO DELAY
    )

    # RUN 2: T+1 (Delayed)
    print("\n>>> RUNNING SIMULATION 2: T+1 Execution (Delayed Entry)")
    eq_t1 = core.run_historical_backtest(
        initial_capital=1000000.0,
        top_n=cfg['alpha_model']['max_positions'],
        energy_threshold=cfg['alpha_model']['energy_threshold'],
        start_date="2015-01-01",
        execution_delay=1  # 1 DAY DELAY
    )

    # Compare
    m0 = calculate_metrics(eq_t0, "T+0 (Standard)")
    m1 = calculate_metrics(eq_t1, "T+1 (Delayed)")
    
    res_df = pd.DataFrame([m0, m1])
    print("\n-------------------------------------------------")
    print("COMPARISON RESULTS")
    print("-------------------------------------------------")
    print(res_df.to_string(index=False))
    print("-------------------------------------------------")
    
    print("\nObservations:")
    rows0 = eq_t0.height
    rows1 = eq_t1.height
    print(f"Days Simulated: {rows0}")

if __name__ == "__main__":
    run()
