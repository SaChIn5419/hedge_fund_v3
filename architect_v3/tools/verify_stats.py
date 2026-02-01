import yaml
import polars as pl
import numpy as np
import os
import sys

# Add parent directory to path to allow importing modules from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.alpha_v1 import DualEngineAlpha
from engine.core_physics import VectorizedCore

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run():
    print("ARCHITECT: Verifying Market-Regime Logic...")
    cfg = load_config()

    # Initialise components
    alpha = DualEngineAlpha(momentum_window=cfg['alpha_model']['momentum_window'])
    core = VectorizedCore(cfg['universe']['nse_parquet_path'], alpha)

    # Run historical backtest
    equity_curve = core.run_historical_backtest(
        initial_capital=1000000.0,
        top_n=cfg['alpha_model']['max_positions'],
        energy_threshold=cfg['alpha_model']['energy_threshold'],
        start_date="2015-01-01"
    )

    # Calculate Metrics manually for console verification
    df = equity_curve.with_columns([
        pl.col("equity").pct_change().alias("ret"),
        pl.col("equity").cum_max().alias("peak")
    ]).with_columns([
        (pl.col("equity") / pl.col("peak") - 1).alias("drawdown")
    ]).drop_nulls()

    total_days = df.height
    years = total_days / 252.0
    cagr = (df["equity"][-1] / df["equity"][0]) ** (1 / years) - 1
    max_dd = df["drawdown"].min()
    final_equity = df["equity"][-1]

    print("\n" + "="*40)
    print(f"RESULTS WITH NIFTY 200-SMA FILTER")
    print("="*40)
    print(f"Final Equity:  ₹ {final_equity:,.2f}")
    print(f"CAGR:          {cagr*100:.2f}%")
    print(f"Max Drawdown:  {max_dd*100:.2f}%")
    print(f"Total Return:  {((final_equity/1000000)-1)*100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    run()
