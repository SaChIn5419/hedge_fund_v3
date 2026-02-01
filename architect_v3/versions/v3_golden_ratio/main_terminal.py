import yaml
from strategies.alpha_v1 import DualEngineAlpha
from engine.core_physics import VectorizedCore
from engine.analytics import PolarsTearSheet

# Terminal colour constants
PRIME = "\033[92m"
ALERT = "\033[91m"
RESET = "\033[0m"

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run():
    print(f"{PRIME}ARCHITECT V3 Online. Systems Normal.{RESET}")
    print("Vector: HISTORICAL INSTITUTIONAL BACKTEST (2015-Present)")
    cfg = load_config()

    # Initialise components
    alpha = DualEngineAlpha(momentum_window=cfg['alpha_model']['momentum_window'])
    core = VectorizedCore(cfg['universe']['nse_parquet_path'], alpha)
    analytics = PolarsTearSheet(risk_free_rate=cfg['physics']['risk_free_rate'])

    # Run historical backtest starting from 2015
    equity_curve = core.run_historical_backtest(
        initial_capital=1000000.0,
        top_n=cfg['alpha_model']['max_positions'],
        energy_threshold=cfg['alpha_model']['energy_threshold'],
        start_date="2015-01-01"
    )

    # Generate tear-sheet with trade data
    analytics.generate(equity_curve, trades_df=core.trades_df)

if __name__ == "__main__":
    run()
