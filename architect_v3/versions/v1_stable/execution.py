import numpy as np

class RiskManager:
    def __init__(self, total_capital, cash_buffer=0.05):
        self.available_cash = total_capital * (1 - cash_buffer)

    def calculate_sizes(self, signals_df):
        if signals_df.height == 0:
            return []

        tickers = signals_df["ticker"].to_list()
        prices = signals_df["close"].to_numpy()
        vols = signals_df["volatility"].to_numpy()

        # Handle zero volatility (Safety Valve)
        vols = np.where(vols == 0, 0.01, vols)

        # Inverse Volatility Weighting
        inv_vols = 1.0 / vols
        weights = inv_vols / np.sum(inv_vols)

        # Calculate exact quantities
        cash_allocations = self.available_cash * weights
        quantities = np.floor(cash_allocations / prices).astype(int)

        orders = []
        for i in range(len(tickers)):
            if quantities[i] > 0:
                orders.append({
                    "ticker": tickers[i],
                    "action": "BUY",
                    "price": prices[i],
                    "quantity": quantities[i]
                })
        return orders
