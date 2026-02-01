# strategies/architect/regime_engine.py
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

class RegimeDetector:
    def __init__(self):
        # Increased n_iter for convergence
        self.model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
        
    def fit_predict(self, df):
        df = df.rename(columns=str.lower)
        
        returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        # Daily Volatility over 20 days (approx 1 trading month)
        volatility = returns.rolling(20).std().fillna(0) 
        kinetic_energy = (df['volume'] * returns.abs()).rolling(10).mean().fillna(0)
        
        # Combine features
        features_df = pd.DataFrame({
            'returns': returns, 
            'vol': volatility, 
            'energy': kinetic_energy
        }).replace([np.inf, -np.inf], 0).fillna(0)
        
        features = features_df.values
        
        if len(features) < 50:
            return 0 # Default to Trend
            
        try:
            self.model.fit(features)
            states = self.model.predict(features)
            
            # Map states based on volatility variance to be safe
            # State 0 should be Low Volatility
            variances = [self.model.covars_[i][1][1] for i in range(3)]
            state_mapping = np.argsort(variances)
            
            current_raw_state = states[-1]
            mapped_state = np.where(state_mapping == current_raw_state)[0][0]
            
            return mapped_state
            
        except Exception:
            return 0
