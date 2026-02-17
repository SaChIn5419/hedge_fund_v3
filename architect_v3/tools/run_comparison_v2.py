import yaml
import sys
import os
import polars as pl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.alpha_v1 import DualEngineAlpha
from engine.core_physics import VectorizedCore
from engine.analytics import PolarsTearSheet

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run():
    print("ARCHITECT V3: FULL STRATEGY COMPARISON (T+0 vs T+1)")
    cfg = load_config()

    alpha = DualEngineAlpha(momentum_window=cfg['alpha_model']['momentum_window'])
    core = VectorizedCore(cfg['universe']['nse_parquet_path'], alpha)
    analytics = PolarsTearSheet(
        risk_free_rate=cfg['physics']['risk_free_rate'],
        benchmark_ticker=cfg['brokers']['dhan'].get('BENCHMARK_TICKER', '^NSEI')
    )
    
    # 1. T+0 BACKTEST
    print("\n>>> EXECUTING T+0 (Standard)...")
    eq_t0 = core.run_historical_backtest(
        initial_capital=1000000.0,
        top_n=cfg['alpha_model']['max_positions'],
        energy_threshold=cfg['alpha_model']['energy_threshold'],
        start_date="2015-01-01",
        execution_delay=0,
        log_filename="trade_log_t0.csv"
    )
    trades_t0 = core.trades_df

    # 2. T+1 BACKTEST
    print("\n>>> EXECUTING T+1 (Delayed)...")
    eq_t1 = core.run_historical_backtest(
        initial_capital=1000000.0,
        top_n=cfg['alpha_model']['max_positions'],
        energy_threshold=cfg['alpha_model']['energy_threshold'],
        start_date="2015-01-01",
        execution_delay=1,
        log_filename="trade_log_t1.csv"
    )
    trades_t1 = core.trades_df

    # 3. GENERATE COMPARISON TEARSHEET
    analytics.generate_comparison(eq_t0, eq_t1, trades_t0, trades_t1)

if __name__ == "__main__":
    run()
