import numpy as np
import pandas as pd
from scipy.stats import entropy

class RegimeDriftDetector:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.returns_buffer = []
        self.current_regime = "STABLE"
        self.entropy_history = []

    def on_tick(self, price):
        """
        Ingests price, updates returns buffer, calculates entropy.
        """
        if not self.returns_buffer:
            self.returns_buffer.append(price) # Store raw price first to get diff later
            return "INITIALIZING"

        # Calculate return from last price
        last_price = self.returns_buffer[-1]
        
        # We store returns strictly after the first price
        if len(self.returns_buffer) == 1:
             # Just append the second price to calculate return next time? 
             # Actually, let's store prices to calc rolling returns safely
             pass
        
        # Update buffer with PRICE (easier to manage)
        self.returns_buffer.append(price)
        
        if len(self.returns_buffer) > self.window_size + 1:
            self.returns_buffer.pop(0)

        # Need enough data to calculate distribution
        if len(self.returns_buffer) < 30:
            return "INITIALIZING"

        # Calculate Returns
        prices = np.array(self.returns_buffer)
        returns = np.diff(prices) / prices[:-1]
        
        # 1. Calculate Entropy of the distribution of returns
        # We histogram the returns to get probabilities
        hist, bins = np.histogram(returns, bins=10, density=True)
        # Normalize histogram to sum to 1 to represent probabilities
        probs = hist / np.sum(hist)
        
        # Shannon Entropy
        ent = entropy(probs)
        self.entropy_history.append(ent)
        
        # 2. Determine Regime
        # High Entropy = Chaos/Noise (Random Walk)
        # Low Entropy = Ordered (Trend)
        
        # Thresholds (Arbitrary for now, need tuning)
        if ent > 2.0: 
            self.current_regime = "CHAOS"
        elif ent < 1.0:
            self.current_regime = "TRENDING"
        else:
            self.current_regime = "STABLE"
            
        return {
            "regime": self.current_regime,
            "entropy": ent,
            "volatility": np.std(returns)
        }
