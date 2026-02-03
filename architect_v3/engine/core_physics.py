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
        # 1. EXTRACT TRIANGULATION SENSORS (Nifty 50, Bank Nifty, VIX)
        # ---------------------------------------------------------
        
        # SENSOR 1: BROAD MARKET (Anchor) - Nifty 50
        broad_query = f"""
            SELECT Date as date, Close as broad_close 
            FROM read_parquet('{self.data_path}', filename=true) 
            WHERE REGEXP_MATCHES(filename, '^NSEI\\.parquet')
            AND Date >= '{start_date}'
            AND Date <= '{end_date}'
        """
        
        # SENSOR 2: RISK CANARY (High Beta) - Bank Nifty
        canary_query = f"""
            SELECT Date as date, Close as canary_close 
            FROM read_parquet('{self.data_path}', filename=true) 
            WHERE REGEXP_MATCHES(filename, '^NSEBANK\\.parquet')
            AND Date >= '{start_date}'
            AND Date <= '{end_date}'
        """
        
        # SENSOR 3: FEAR SIREN (VIX)
        vix_query = f"""
            SELECT Date as date, Close as vix_close
            FROM read_parquet('{self.data_path}', filename=true)
            WHERE REGEXP_MATCHES(filename, '^INDIAVIX\\.parquet')
            AND Date >= '{start_date}'
            AND Date <= '{end_date}'
        """

        try:
            broad = duckdb.sql(broad_query).pl().with_columns(pl.col("date").cast(pl.Datetime("us")))
            canary = duckdb.sql(canary_query).pl().with_columns(pl.col("date").cast(pl.Datetime("us")))
            try:
                vix = duckdb.sql(vix_query).pl().with_columns(pl.col("date").cast(pl.Datetime("us")))
            except:
                vix = None

            # Join Sensors
            sensors = broad.join(canary, on="date", how="left").sort("date")
            if vix is not None:
                sensors = sensors.join(vix, on="date", how="left").fill_null(0.0)
            else:
                sensors = sensors.with_columns(pl.lit(0.0).alias("vix_close"))
                
            # Calculate Indicators
            sensors = sensors.with_columns([
                # BROAD: SMA 200 (Result: Bear if < SMA200)
                pl.col("broad_close").rolling_mean(window_size=200).shift(1).alias("broad_sma_200"),
                
                # CANARY: SMA 50 + ROC 20
                pl.col("canary_close").rolling_mean(window_size=50).shift(1).alias("canary_sma_50"),
                (pl.col("canary_close") / pl.col("canary_close").shift(21) - 1).alias("canary_roc_20")
            ]).with_columns([
                # TRIANGULATION LOGIC:
                # 1. Broad Trend Broken?
                (pl.col("broad_close") < pl.col("broad_sma_200")).alias("signal_broad_bear"),
                
                # 2. Canary Trend Broken or Crashing?
                ((pl.col("canary_close") < pl.col("canary_sma_50")) | (pl.col("canary_roc_20") < -0.05)).alias("signal_canary_bear"),
                
                # 3. Panic Spike?
                (pl.col("vix_close") > 24.0).alias("signal_vix_panic")
            ]).with_columns([
                # FINAL VOTE: If ANY are True -> BEAR
                pl.when(
                    pl.col("signal_broad_bear") | pl.col("signal_canary_bear") | pl.col("signal_vix_panic")
                )
                .then(0.0)
                .otherwise(1.0)
                .alias("market_regime"),
                
                # Pass VIX for display
                (pl.col("vix_close") / 100.0).alias("nifty_vol"),
                
                # Dummy scaler
                pl.lit(1.0).alias("vol_scaler")
                
            ]).select(["date", "market_regime", "vol_scaler", "nifty_vol"])
            
            nifty = sensors # Alias for downstream compatibility

        except Exception as e:
            print(f"ARCHITECT WARNING: Could not load Sensor data ({e}). Defaulting to Bull Market.")
            base_dates = self._get_unified_lazyframe(start_date).select("date").unique().collect()
            nifty = base_dates.with_columns([
                pl.lit(1.0).alias("market_regime"),
                pl.lit(1.0).alias("vol_scaler"),
                pl.lit(0.15).alias("nifty_vol")
            ])
        
        # ---------------------------------------------------------
        # 2. RUN STANDARD STRATEGY
        # ---------------------------------------------------------
        base = self._get_unified_lazyframe(start_date=start_date)

        # Join Regime EARLY (To scale weights)
        # Ensure date columns maximize compatibility (cast to Datetime[us])
        base = base.with_columns(pl.col("date").cast(pl.Datetime("us")))
        base = base.sort("date").join(nifty.lazy(), on="date", how="left").fill_null(1.0)

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
        
        # --- DATA HYGIENE: FILTER ANOMALIES ---
        # Exclude stocks with > 5% Daily Volatility (likely Penny Stocks or Splits)
        # Exclude stocks with > 300% Momentum (likely Pump & Dump or Data Error)
        filtered = filtered.filter(pl.col("volatility") < 0.05)
        filtered = filtered.filter(pl.col("momentum") < 3.0) # Cap at 300%
        
        # --- PHASE 8: AMT / VOLUME PROFILE PROXY ---
        # "Avoid The Chop"
        # Efficiency Ratio > 0.3 means price is moving with purpose (Out of Value).
        # Efficiency Ratio <= 0.3 means prices are noise (In Value/Equilibrium).
        filtered = filtered.filter(pl.col("efficiency") > 0.3)

        trades_lf = (
            filtered.sort(["date", "momentum"], descending=[False, True])
            .group_by("date").head(top_n)
            
            # --- ARCHITECT PROTOCOL V7 (The Regime Guard) ---
            # 1. SYSTEM STATE (Derived from Nifty)
            #    market_regime is already joined: 1.0 (Bull), 0.5 (Vol), 0.0 (Bear)
            
            # 2. ALPHA ENGINE (Aggressive Risk Parity)
            #    Target Vol = 30% (High Conviction)
            .with_columns([
                 (pl.col("volatility") * (252**0.5)).alias("ann_vol")
            ])
            .with_columns([
                (0.30 / pl.col("ann_vol")).alias("alpha_weight")
            ])

            # 3. Momentum Scaler (Size up winners)
            .with_columns([
                pl.col("momentum").clip(0.0, 1.0).alias("signal_strength")
            ])

            # 4. Apply Regime Guard
            #    Final Weight = Alpha * Signal * Regime
            .with_columns([
                (pl.col("rupee_volume") * MAX_VOLUME_PARTICIPATION / initial_capital).alias("max_weight")
            ])
            .with_columns([
                # Constraints: Cap 20% (Max 1.0x Leverage across 5 assets)
                # REVERTED: Removed all Dynamic Leverage / Vol Targeting.
                # Logic: Risk Parity * Momentum * Regime Guard.
                pl.min_horizontal(
                    (
                        pl.col("alpha_weight") * 
                        pl.col("signal_strength") * 
                        pl.col("market_regime")
                    ),
                    pl.col("max_weight"), 
                    pl.lit(0.20)
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
            (pl.col("net_return") * initial_capital).alias("net_pnl"),
            
            # --- BLACK BOX RECORDER LOGIC (Vectorized) ---
            # 1. Structure Tag
            pl.when(pl.col("efficiency") > 0.3).then(pl.lit("VACUUM")).otherwise(pl.lit("TRAPPED")).alias("structure_tag"),
            
            # 2. Market State
            pl.when(pl.col("market_regime") == 1.0).then(pl.lit("BULL_TURBO"))
              .when(pl.col("market_regime") == 0.5).then(pl.lit("VOLATILE_CRUISE"))
              .otherwise(pl.lit("BEAR_DEFENSE")).alias("market_state"),
              
            # 3. Decision Reason (Explainability Layer)
            pl.when(pl.col("market_regime") == 0.0).then(pl.lit("Regime Guard (Bear)"))
              .when(pl.col("market_regime") == 0.5).then(pl.lit("Vol Penalty (Risk Off)"))
              .when(pl.col("efficiency") < 0.3).then(pl.lit("Volume Profile (Chop)")) # Won't trigger if filtered
              .when(pl.col("kinetic_energy") > 0.0025).then(pl.lit("High Energy Trend"))
              .otherwise(pl.lit("Standard Entry")).alias("decision_reason"),
            
            # 4. Math: Leverage Mult (Fixed at 1.0 for Pure Cash)
            pl.lit(1.0).alias("leverage_mult"),
            
            # 5. Math: Gaussian Z-Score approx
            # Width = 4 std devs (2 up, 2 down). Std = (Upper - Lower) / 4
            ((pl.col("close") - pl.col("sma_20")) / ((pl.col("upper_band") - pl.col("lower_band")) / 4)).alias("gaussian_zscore")
        ]).select([
            "date", "ticker", "close", "sma_20", "upper_band", "lower_band",
            "momentum", "energy", "volatility", "rupee_volume",
            "alpha_weight", "market_regime", "max_weight", "weight", "fwd_return", "turnover",
            "position_value", "gross_pnl", "friction_pnl", "net_pnl",
            "kinetic_energy", "efficiency", "nifty_vol",
            # --- BLACK BOX RECORDER COLUMNS ---
            "structure_tag", "leverage_mult", "market_state", "decision_reason", "gaussian_zscore"
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
        
        # 4. APPLY LATE REGIME FILTER (Removed - Handled at Weight Level)
        # Note: We already applied the regime to the weights.
        # So 'portfolio_return' doesn't need a second multiplier.
        daily_portfolio = daily_portfolio.with_columns(
             pl.col("raw_portfolio_return").alias("portfolio_return")
        )

        # 5. COMPOUND EQUITY
        equity_curve = (
            daily_portfolio.with_columns([
                (pl.lit(initial_capital) * (1 + pl.col("portfolio_return")).cum_prod()).alias("equity")
            ])
        )

        elapsed = time.time() - start_time
        print(f"ARCHITECT: Backtest completed in {elapsed:.4f} seconds.")
        return equity_curve
