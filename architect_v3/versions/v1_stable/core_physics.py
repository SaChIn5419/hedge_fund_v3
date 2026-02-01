import polars as pl
import duckdb
import time
from datetime import datetime

class VectorizedCore:
    def __init__(self, data_path: str, alpha_model):
        self.data_path = data_path
        self.alpha_model = alpha_model

    def _get_unified_lazyframe(self, start_date: str = "2015-01-01"):
        """
        The DuckDB Ingestion Bridge.
        Excludes Indices and filters from start_date.
        """
        query = f"""
            SELECT 
                Date as date, 
                Close as close, 
                High as high, 
                Low as low, 
                Open as open, 
                Volume as volume,
                REGEXP_EXTRACT(filename, '([^/\\\\\\\\]+)\\.parquet$', 1) as ticker
            FROM read_parquet('{self.data_path}', filename=true)
            WHERE Close IS NOT NULL 
            AND Date >= '{start_date}'
            AND NOT REGEXP_MATCHES(filename, '\\^')
        """
        return duckdb.sql(query).pl().lazy()

    def generate_live_signals(self, energy_threshold=0.05, top_n=5):
        start_time = time.time()
        lf = self._get_unified_lazyframe()

        q_engine = (
            lf.sort(["ticker", "date"])
            .with_columns(self.alpha_model.get_expressions())
            .with_columns([(pl.col("close") * pl.col("volume")).alias("rupee_volume")])
            .filter(pl.col("date") == pl.col("date").max())
            .filter(pl.col("rupee_volume") > 50000000)
            .filter(pl.col("energy") >= energy_threshold)
            .drop_nulls()
            .sort("momentum", descending=True)
            .head(top_n)
        )

        signals = q_engine.collect()
        elapsed = time.time() - start_time
        print(f"ARCHITECT: Matrix Resolved in {elapsed:.4f} seconds.")
        return signals

    def run_historical_backtest(self, initial_capital=1000000.0, top_n=5, energy_threshold=0.05, start_date="2015-01-01"):
        print(f"ARCHITECT: Initializing Reality-Capped Historical Engine (from {start_date})...")
        start_time = time.time()

        # REALITY CONSTANTS
        SLIPPAGE_RATE = 0.0050
        MAX_VOLUME_PARTICIPATION = 0.02

        base = self._get_unified_lazyframe(start_date=start_date)

        # Add Gaussian Channel indicators for visualization
        strategy = (
            base.sort(["ticker", "date"])
            .with_columns(self.alpha_model.get_expressions())
            .with_columns([
                # Gaussian Channel: SMA as the center, with upper/lower bands
                pl.col("close").rolling_mean(window_size=20).over("ticker").alias("sma_20"),
                pl.col("close").rolling_std(window_size=20).over("ticker").alias("std_20"),
            ])
            .with_columns([
                (pl.col("sma_20") + 2 * pl.col("std_20")).alias("upper_band"),
                (pl.col("sma_20") - 2 * pl.col("std_20")).alias("lower_band"),
                (pl.col("close").shift(-1) / pl.col("close") - 1).over("ticker").alias("fwd_return"),
                (pl.col("close") * pl.col("volume")).alias("rupee_volume")
            ])
            .drop_nulls()
        )

        # Filter illiquid stocks
        filtered = strategy.filter(pl.col("rupee_volume") > 50000000)
        filtered = filtered.filter(pl.col("energy") >= energy_threshold)

        # Risk Parity Sizing + Liquidity Capping
        trades_lf = (
            filtered.sort(["date", "momentum"], descending=[False, True])
            .group_by("date").head(top_n)
            .with_columns([(1.0 / pl.col("volatility")).alias("inv_vol")])
            .with_columns([
                ((pl.col("inv_vol") / pl.col("inv_vol").sum()) * 0.95).over("date").alias("base_weight")
            ])
            .with_columns([
                (pl.col("rupee_volume") * MAX_VOLUME_PARTICIPATION / initial_capital).alias("max_weight")
            ])
            .with_columns([
                pl.min_horizontal("base_weight", "max_weight").alias("weight")
            ])
            .sort(["ticker", "date"])
            .with_columns([
                (pl.col("weight").diff().fill_null(pl.col("weight")).abs()).over("ticker").alias("turnover")
            ])
            .with_columns([
                (pl.col("weight") * pl.col("fwd_return")).alias("gross_return"),
                (pl.col("turnover") * SLIPPAGE_RATE).alias("friction_cost"),
                ((pl.col("weight") * pl.col("fwd_return")) - (pl.col("turnover") * SLIPPAGE_RATE)).alias("net_return")
            ])
        )

        # Collect trades
        trades_df = trades_lf.collect()

        # Calculate P&L in rupees
        trades_df = trades_df.with_columns([
            (pl.col("weight") * initial_capital).alias("position_value"),
            (pl.col("gross_return") * initial_capital).alias("gross_pnl"),
            (pl.col("friction_cost") * initial_capital).alias("friction_pnl"),
            (pl.col("net_return") * initial_capital).alias("net_pnl")
        ])

        # Export trade log with Gaussian channel data
        trade_log = trades_df.select([
            "date", "ticker", "close", "sma_20", "upper_band", "lower_band",
            "momentum", "energy", "volatility", "rupee_volume",
            "base_weight", "max_weight", "weight", "fwd_return", "turnover",
            "position_value", "gross_pnl", "friction_pnl", "net_pnl"
        ]).sort(["date", "ticker"])
        
        trade_log.write_csv("trade_log.csv")
        print(f"ARCHITECT: Trade log exported ({trade_log.height} trades)")

        # Aggregate to Portfolio level
        daily_portfolio = (
            trades_df.group_by("date")
            .agg(pl.col("net_return").sum().alias("portfolio_return"))
            .sort("date")
        )

        # Compound the Equity Curve
        equity_curve = daily_portfolio.with_columns([
            (pl.lit(initial_capital) * (1 + pl.col("portfolio_return")).cum_prod()).alias("equity")
        ])

        elapsed = time.time() - start_time
        print(f"ARCHITECT: Backtest completed in {elapsed:.4f} seconds.")
        
        # Store trades for visualization
        self.trades_df = trades_df
        
        return equity_curve
