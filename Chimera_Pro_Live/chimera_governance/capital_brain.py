class CapitalAllocationBrain:
    def __init__(self, base_leverage=1.0):
        self.base_leverage = base_leverage
        self.current_leverage = base_leverage

    def update_allocation(self, regime_data):
        """
        Adjusts leverage based on Drift Detector output.
        """
        regime = regime_data.get("regime", "STABLE")
        vol = regime_data.get("volatility", 0.01)
        
        if regime == "CHAOS":
            # High Entropy / Noise -> Defensive
            target = 0.5
        elif regime == "TRENDING":
            # Low Entropy -> Conviction
            target = 1.5
        else:
            # STABLE
            target = 1.0
        
        # Volatility Scaling (Safety Net)
        if vol > 0.03: # 3% volatility is high
            target = min(target, 0.5)
            
        self.current_leverage = target
        return self.current_leverage
