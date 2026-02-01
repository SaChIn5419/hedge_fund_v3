import polars as pl
import duckdb
import time
from datetime import datetime

class VectorizedCore:
    def __init__(self, data_path: str, alpha_model):
        self.data_path = data_path
        self.alpha_model = alpha_model

    def _get_unified_lazyframe(self, start_date: str = "2015-01-01", end_date: str = "2099-12-31"):
        """
        The DuckDB Ingestion Bridge.
        Excludes Indices and filters from start_date to end_date.
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
            AND Date <= '{end_date}'
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

    def run_historical_backtest(self, initial_capital=1000000.0, top_n=5, energy_threshold=0.05, start_date="2015-01-01", end_date="2099-12-31"):
        print(f"ARCHITECT: Initializing Market-Regime Backtest ({start_date} to {end_date})...")
        start_time = time.time()

        # REALITY CONSTANTS (Updated Feb 2026)
        SLIPPAGE_RATE = 0.0032 
        MAX_VOLUME_PARTICIPATION = 0.02

        # ---------------------------------------------------------
        # 1. EXTRACT NIFTY 50 REGIME (The Master Circuit Breaker)
        # ---------------------------------------------------------
        nifty_query = f"""
            SELECT Date as date, Close as close 
            FROM read_parquet('{self.data_path}', filename=true) 
            WHERE REGEXP_MATCHES(filename, '\\^NSEI\\.parquet')
            AND Date >= '{start_date}'
            AND Date <= '{end_date}'
        """
        try:
            nifty = duckdb.sql(nifty_query).pl()
            
            # Calculate 200-Day SMA for the Nifty
            nifty = nifty.with_columns(
                pl.col("date").cast(pl.Datetime("us")).alias("date")
            ).sort("date").with_columns([
                pl.col("close").rolling_mean(window_size=200).shift(1).alias("nifty_sma_200")
            ]).with_columns([
                # 1 = Bull Market (Buy), 0 = Bear Market (Cash)
                pl.when(pl.col("close") > pl.col("nifty_sma_200"))
                .then(1.0).otherwise(0.0).alias("market_regime")
            ]).select(["date", "market_regime"])
        except Exception as e:
            print(f"ARCHITECT WARNING: Could not load Nifty data for regime filter ({e}). Defaulting to Bull Market.")
            base_dates = self._get_unified_lazyframe(start_date).select("date").unique().collect()
            nifty = base_dates.with_columns(pl.lit(1.0).alias("market_regime"))

        # ---------------------------------------------------------
        # 2. RUN STANDARD STRATEGY (NO EARLY FILTER)
        # ---------------------------------------------------------
        base = self._get_unified_lazyframe(start_date=start_date)

        strategy = (
            base.sort(["ticker", "date"])
            .with_columns(self.alpha_model.get_expressions())
            .with_columns([
                pl.col("close").rolling_mean(window_size=20).over("ticker").alias("sma_20"),
                pl.col("close").rolling_std(window_size=20).over("ticker").alias("std_20"),
            ])
            .with_columns([
                (pl.col("sma_20") + 2 * pl.col("std_20")).alias("upper_band"),
                (pl.col("sma_20") - 2 * pl.col("std_20")).alias("lower_band"),
                # CLOSE-CLOSE EXECUTION: Captures Overnight Gaps + Intraday Momentum.
                # Combined with Trailing Stop-Loss for downside protection.
                (pl.col("close").shift(-1) / pl.col("close") - 1).over("ticker").alias("fwd_return"),
                (pl.col("close") * pl.col("volume")).alias("rupee_volume")
            ])
            .drop_nulls()
        )
        
        # NOTE: Early filter removed to match the high-profit logic.
        # We process ALL regimes, then zero out returns later.

        # ---------------------------------------------------------
        # 3. APPLY TRAILING STOP-LOSS (The "Chandelier Exit")
        # ---------------------------------------------------------
        # Vectorized Approximation:
        # If Close is > 15% below the 60-Day High, we assume the trend is broken and EXIT.
        # This prevents holding "falling knives" during crash phases.
        strategy = strategy.with_columns([
            pl.col("close").rolling_max(window_size=60).over("ticker").alias("rolling_peak_60")
        ]).filter(
            pl.col("close") > (pl.col("rolling_peak_60") * 0.85)
        )

        filtered = strategy.filter(pl.col("rupee_volume") > 50000000)
        filtered = filtered.filter(pl.col("energy") >= energy_threshold)

        trades_lf = (
            filtered.sort(["date", "momentum"], descending=[False, True])
            .group_by("date").head(top_n)
            
            # --- ARCHITECT PROTOCOL v5.0 (Intelligent Velocity) ---
            # 1. METRICS: 20-Day Volatility & 50-Day SMA
            .with_columns([
                 (pl.col("volatility") * (252**0.5)).alias("ann_vol"),
                 pl.col("close").rolling_mean(window_size=50).over("ticker").alias("sma_50")
            ])
            .with_columns([
                # 2. BASE SIZING (Risk Parity, Target Vol = 30%)
                #    Weight = 0.30 / Ann_Vol
                (0.30 / pl.col("ann_vol")).alias("raw_weight")
            ])
            .with_columns([
                # 3. INTELLIGENT BRAKE LOGIC
                #    Condition A: Extreme Volatility (> 80%)
                #    Condition B: Downtrend (Price < SMA 50)
                #    Logic: Only Brake (0.33x) if BOTH are true. Else, Full Speed.
                pl.when((pl.col("ann_vol") > 0.80) & (pl.col("close") < pl.col("sma_50")))
                .then(pl.col("raw_weight") * 0.33)
                .otherwise(pl.col("raw_weight"))
                .alias("vol_weight")
            ])
            
            # 4. Momentum Scaler (Retained from v1.0)
            .with_columns([
                pl.col("momentum").clip(0.0, 1.0).alias("signal_strength")
            ])

            .with_columns([
                (pl.col("rupee_volume") * MAX_VOLUME_PARTICIPATION / initial_capital).alias("max_weight")
            ])
            .with_columns([
                # Final Weight Calculation
                # Constraints: 
                # 1. Cap single stock at 30% (0.30) - Increased for v5
                # 2. Liquidity (Max Weight)
                # 3. Intelligent Volatility Sizing
                pl.min_horizontal(
                    (pl.col("vol_weight") * pl.col("signal_strength")),
                    pl.col("max_weight"), 
                    pl.lit(0.30)  # Protocol v5: 30% Max
                ).alias("weight")
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

        trades_df = trades_lf.collect()

        # Export trade log
        trade_log = trades_df.with_columns([
            (pl.col("weight") * initial_capital).alias("position_value"),
            (pl.col("gross_return") * initial_capital).alias("gross_pnl"),
            (pl.col("friction_cost") * initial_capital).alias("friction_pnl"),
            (pl.col("net_return") * initial_capital).alias("net_pnl")
        ]).select([
            "date", "ticker", "close", "sma_20", "upper_band", "lower_band",
            "momentum", "energy", "volatility", "rupee_volume",
            "vol_weight", "max_weight", "weight", "fwd_return", "turnover",
            "position_value", "gross_pnl", "friction_pnl", "net_pnl"
        ]).sort(["date", "ticker"])
        
        trade_log.write_csv("trade_log.csv")
        self.trades_df = trade_log
        print(f"ARCHITECT: Trade log exported ({trade_log.height} trades)")

        # 3. AGGREGATE TO PORTFOLIO
        daily_portfolio = (
            trades_df.group_by("date")
            .agg(pl.col("net_return").sum().alias("raw_portfolio_return"))
            .sort("date")
        )
        
        # 4. APPLY LATE REGIME FILTER (Portfolio Level)
        # Join the Nifty Regime with our daily portfolio returns
        # Join on datetime[us] to match Nifty (which we cast to us earlier)
        daily_portfolio = daily_portfolio.with_columns(
            pl.col("date").cast(pl.Datetime("us"))
        ).join(nifty, on="date", how="left").fill_null(1.0)

        # Apply the Circuit Breaker: If Nifty is bearish, return is 0 (Cash)
        daily_portfolio = daily_portfolio.with_columns([
            (pl.col("raw_portfolio_return") * pl.col("market_regime")).alias("portfolio_return")
        ])

        # 5. COMPOUND EQUITY
        equity_curve = (
            daily_portfolio.with_columns([
                (pl.lit(initial_capital) * (1 + pl.col("portfolio_return")).cum_prod()).alias("equity")
            ])
        )

        elapsed = time.time() - start_time
        print(f"ARCHITECT: Backtest completed in {elapsed:.4f} seconds.")
        return equity_curve
