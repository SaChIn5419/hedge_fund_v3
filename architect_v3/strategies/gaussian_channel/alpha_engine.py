# strategies/architect/alpha_engine.py
import numpy as np
import pandas as pd
import pandas_ta as ta

class SignalProcessor:
    @staticmethod
    def kalman_filter(prices, Q=0.0005, R=0.097):
        """Removes Brownian Noise from price stream"""
        # Ensure input is a Series, handle potential Series/DataFrame ambiguity
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
            
        n = len(prices)
        xhat = np.zeros(n)
        P = np.zeros(n)
        
        # Initialize
        xhat[0] = prices.iloc[0] if n > 0 else 0
        P[0] = 1.0
        
        for t in range(1, n):
            xhat_prior = xhat[t-1]
            P_prior = P[t-1] + Q
            K = P_prior / (P_prior + R)
            xhat[t] = xhat_prior + K * (prices.iloc[t] - xhat_prior)
            P[t] = (1 - K) * P_prior
            
        return pd.Series(xhat, index=prices.index)

    @staticmethod
    def ehlers_gaussian_channel(df, length=20, mult=1.414):
        """Calculates Zero-Lag Gaussian Channel with Filtered True Range"""
        # 4-Pole Gaussian Filter approximation for speed
        alpha = 2 / (length + 1)
        beta = 1 - alpha
        
        # Ensure we are working with Series
        src = df['close'] if 'close' in df.columns else df['Close']
        if isinstance(src, pd.DataFrame): src = src.iloc[:, 0]

        filt = np.zeros(len(src))
        
        # Need at least 5 points
        if len(src) < 5:
            df['Gaussian_Filt'] = src
            df['Gaussian_Upper'] = src
            df['Gaussian_Lower'] = src
            return df
            
        # Initialize first few points to avoid zeros
        filt[0:4] = src.iloc[0:4]
        
        src_values = src.values
        for i in range(4, len(src)):
            filt[i] = (alpha**4)*src_values[i] + 4*beta*filt[i-1] - 6*(beta**2)*filt[i-2] + 4*(beta**3)*filt[i-3] - (beta**4)*filt[i-4]
        
        df['Gaussian_Filt'] = filt
        
        # Filtered True Range for Volatility Bands
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        close = df['close'] if 'close' in df.columns else df['Close']
        
        df['TR'] = ta.true_range(high, low, close)
        df['FTR'] = df['TR'].rolling(window=length).mean() # Simplified FTR
        
        df['Gaussian_Upper'] = df['Gaussian_Filt'] + (df['FTR'] * mult)
        df['Gaussian_Lower'] = df['Gaussian_Filt'] - (df['FTR'] * mult)
        
        return df

    @staticmethod
    def generate_signals(df):
        # Normalize columns (Handle Parquet Capitalization)
        df = df.rename(columns=str.lower)
        
        df = SignalProcessor.ehlers_gaussian_channel(df)
        df['Kalman_Price'] = SignalProcessor.kalman_filter(df['close'])
        
        # KINETIC ENERGY FILTER (Daily)
        df['Vol_MA'] = ta.sma(df['volume'], length=50) # 50-Day Avg Volume
        df['High_Energy'] = df['volume'] > (df['Vol_MA'] * 1.5)
        
        df['Bull_Signal'] = np.where((df['close'] > df['Gaussian_Upper']) & df['High_Energy'], 1, 0)
        df['Bear_Signal'] = np.where((df['close'] < df['Gaussian_Lower']) & df['High_Energy'], -1, 0)
        return df
