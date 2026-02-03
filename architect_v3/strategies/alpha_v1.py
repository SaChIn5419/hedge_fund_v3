import polars as pl
import numpy as np

class DualEngineAlpha:
    def __init__(self, momentum_window=90):
        self.window = momentum_window

    def get_expressions(self):
        """
        LAW 1: The Physics Filter.
        MANDATORY: Every indicator ends with .shift(1) to kill Look-Ahead Bias.
        """
        ret = (pl.col("close") / pl.col("close").shift(1) - 1)
        
        return [
            # Momentum: 90-day returns, shifted by 1 day.
            (pl.col("close") / pl.col("close").shift(self.window) - 1).over("ticker").shift(1).alias("momentum"),
            
            # Kinetic Energy: Volume * (Daily Returns)^2, shifted by 1 day.
            (pl.col("volume") * (ret ** 2)).over("ticker").shift(1).alias("energy"),

            # Volatility (Risk): 20-day standard deviation, shifted by 1 day.
            ret.rolling_std(window_size=20).over("ticker").shift(1).alias("volatility"),

            # AMT PROXY: Efficiency Ratio (The "Anti-Chop" Filter)
            # ER = |Net Change| / Sum(|Change|)
            # Measures if price is traveling in a straight line (Breakout) or noise (Value Area).
            (
                (pl.col("close") - pl.col("close").shift(self.window)).abs() / 
                (pl.col("close").diff().abs().rolling_sum(window_size=self.window))
            ).over("ticker").shift(1).fill_null(0.0).alias("efficiency"),

            # PHASE 9: KINETIC ENERGY (Slope Proxy)
            # User requires: Normalized Slope of 20-day regression.
            # Proxy: 20-Day ROC.
            # High Energy (>0.25% daily) ~= >5% monthly.
            # Low Energy (<0.05% daily) ~= <1% monthly.
            (pl.col("close") / pl.col("close").shift(20) - 1).over("ticker").shift(1).alias("kinetic_energy")
        ]

