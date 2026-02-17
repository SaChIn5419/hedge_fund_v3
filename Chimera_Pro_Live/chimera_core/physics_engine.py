import pandas as pd
import numpy as np
from scipy.stats import linregress
from collections import deque
from datetime import datetime
import time

class ChimeraPhysicsEngine:
    def __init__(self, lookback_energy=20, lookback_vp=60, lookback_gaussian=140):
        self.lookback_energy = lookback_energy
        self.lookback_vp = lookback_vp
        self.lookback_gaussian = lookback_gaussian
        
        # Store candles: { token: DataFrame }
        # Structure: index=timestamp, columns=[Open, High, Low, Close, Volume]
        self.candle_buffers = {} 
        
        # Temporary tick storage to build current candle
        # { token: { 'open': p, 'high': p, 'low': p, 'close': p, 'volume': v, 'start_time': t } }
        self.active_candles = {}
        
        # Candle Size in seconds (Default 1 minute for live intraday physics)
        self.candle_interval = 60 

    def on_tick(self, tick):
        """
        Ingests a tick, updates the candle buffer, and returns valid Physics Signals if a candle closed.
        """
        token = tick['symboltoken']
        price = tick['price']
        timestamp = pd.Timestamp.now() # Use local system time for alignment
        
        # Initialize buffer if new
        if token not in self.candle_buffers:
            self.candle_buffers[token] = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            self.active_candles[token] = {
                'open': price, 'high': price, 'low': price, 'close': price, 
                'volume': 0, 'start_time': time.time()
            }

        # Update Active Candle
        candle = self.active_candles[token]
        candle['high'] = max(candle['high'], price)
        candle['low'] = min(candle['low'], price)
        candle['close'] = price
        # Volume info usually comes cumulative in ticks or discrete. 
        # For simple price physics, we might just track tick count or 0 if vol not avail.
        candle['volume'] += 1 

        # Check for Candle Closure (Time-based)
        elapsed = time.time() - candle['start_time']
        
        if elapsed >= self.candle_interval:
            # Close the candle
            new_row = pd.DataFrame([{
                'Open': candle['open'],
                'High': candle['high'],
                'Low': candle['low'],
                'Close': candle['close'],
                'Volume': candle['volume']
            }], index=[timestamp])
            
            # Append to buffer
            if not new_row.dropna().empty: # Ensure new_row is not empty or NA before concat
                self.candle_buffers[token] = pd.concat([self.candle_buffers[token], new_row])
            
            # Trim Buffer (Keep max required history)
            max_lookback = max(self.lookback_energy, self.lookback_vp, self.lookback_gaussian) + 50
            if len(self.candle_buffers[token]) > max_lookback:
                self.candle_buffers[token] = self.candle_buffers[token].iloc[-max_lookback:]
            
            # Reset Active Candle
            self.active_candles[token] = {
                'open': price, 'high': price, 'low': price, 'close': price, 
                'volume': 0, 'start_time': time.time()
            }
            
            # RUN PHYSICS
            return self.calculate_physics(token)
            
        return None

    def calculate_physics(self, token):
        """
        Runs the Econophysics modules on the buffer.
        """
        df = self.candle_buffers[token]
        
        # 1. Kinetic Energy
        energy = self.get_energy(df['Close'])
        
        # 2. Structure (Vacuum)
        structure = self.check_structure(df)
        
        # 3. Gaussian Signal
        signal = self.get_gaussian_signal(df)
        
        return {
            "token": token,
            "energy": energy,
            "structure": structure,
            "signal": signal,
            "timestamp": str(datetime.now())
        }

    # --- PHYSICS KERNELS ---
    
    def get_energy(self, series):
        """Calculates Slope (Kinetic Energy)"""
        if len(series) < self.lookback_energy: return 0.0
        y = series.tail(self.lookback_energy).values
        x = np.arange(len(y))
        slope, _, _, _, _ = linregress(x, y)
        if y.mean() == 0: return 0.0
        return (slope / y.mean()) * 100

    def check_structure(self, df):
        """Volume Profile: Returns VACUUM (True) or TRAPPED (False)"""
        if len(df) < self.lookback_vp: return "N/A"
        
        hist = df.iloc[-(self.lookback_vp+1):-1] 
        current = df['Close'].iloc[-1]
        
        mean = hist['Close'].mean()
        std = hist['Close'].std()
        if std == 0: return "TRAPPED"
        z = (current - mean) / std
        
        if abs(z) < 1.0: return "TRAPPED"
        return "VACUUM"

    def get_gaussian_signal(self, df):
        if len(df) < self.lookback_gaussian: return "NEUTRAL"
        mean = df['Close'].rolling(self.lookback_gaussian).mean().iloc[-1]
        std = df['Close'].rolling(self.lookback_gaussian).std().iloc[-1]
        price = df['Close'].iloc[-1]
        
        if price > mean + 2*std: return "BREAKOUT"
        elif price < mean - 2*std: return "BREAKDOWN"
        return "NEUTRAL"
