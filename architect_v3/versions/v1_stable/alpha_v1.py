import polars as pl

class DualEngineAlpha:
    def __init__(self, momentum_window=90):
        self.window = momentum_window

    def get_expressions(self):
        """
        LAW 1: The Physics Filter.
        MANDATORY: Every indicator ends with .shift(1) to kill Look-Ahead Bias.
        """
        # Close / Close shift 1 - 1
        ret = (pl.col("close") / pl.col("close").shift(1) - 1)
        
        return [
            # Momentum: 90-day returns, shifted by 1 day.
            (pl.col("close") / pl.col("close").shift(self.window) - 1).over("ticker").shift(1).alias("momentum"),
            
            # Kinetic Energy: Volume * (Daily Returns)^2, shifted by 1 day.
            (pl.col("volume") * (ret ** 2)).over("ticker").shift(1).alias("energy"),

            # Volatility (Risk): 20-day standard deviation, shifted by 1 day.
            ret.rolling_std(window_size=20).over("ticker").shift(1).alias("volatility")
        ]
