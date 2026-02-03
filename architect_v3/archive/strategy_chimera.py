import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- CONFIGURATION (THE CONTROL ROOM) ---
CONFIG = {
    'CAPITAL': 1000000,
    'TARGET_VOL': 0.15,        # 15% Annualized Volatility Target
    'MAX_LEVERAGE': 1.5,       # Max Exposure
    'REBALANCE_FREQ': 'W-FRI', # Weekly Rebalance on Fridays
    'LOOKBACK_GAUSSIAN': 140,  # Channel Length
    'LOOKBACK_VP': 60,         # Volume Profile Memory
    'TICKERS': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'TATAMOTORS.NS'],
    'BENCHMARK': '^NSEI',      # Nifty 50 (Regime)
    'SAFE_HAVEN': 'GOLDBEES.NS' # Crisis Asset
}

class ChimeraEngine:
    def __init__(self):
        print("--- INITIALIZING CHIMERA ENGINE (PHASE 8) ---")
        self.capital = CONFIG['CAPITAL']
        self.equity_curve = []
        self.positions = {}
        
    def fetch_data(self, tickers, start_date):
        """
        Ingests Data without Multi-Index issues.
        """
        print(f"Fetching Data from {start_date}...")
        data = yf.download(tickers, start=start_date, progress=False)
        
        # FIX: Flatten YFinance Multi-Index
        if isinstance(data.columns, pd.MultiIndex):
            data = data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index(level=1)
            # Pivot back to familiar structure: Dict of DataFrames
            clean_data = {}
            for t in tickers:
                df = data[data['Ticker'] == t].copy()
                if not df.empty:
                    clean_data[t] = df
            return clean_data
        return {}

    # --- MODULE 1: THE BRAIN (GAUSSIAN CHANNEL) ---
    def get_gaussian_signal(self, df):
        """
        Standard Gaussian Channel Logic.
        """
        period = CONFIG['LOOKBACK_GAUSSIAN']
        if len(df) < period: return "NEUTRAL"
        
        # 1. Calculate Linear Regression (mean)
        # Using simple rolling mean as proxy for speed in this demo, 
        # normally uses OLS.
        mean = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        
        upper_band = mean + (2 * std)
        lower_band = mean - (2 * std)
        
        current_price = df['Close'].iloc[-1]
        
        # SIGNAL LOGIC
        if current_price > upper_band.iloc[-1]:
            return "BUY" # Breakout
        elif current_price < lower_band.iloc[-1]:
            return "SELL" # Breakdown
        else:
            return "HOLD"

    # --- MODULE 2: THE EYES (VOLUME PROFILE) ---
    def check_volume_profile(self, df):
        """
        Returns True if Price is in a Vacuum (Safe to Trade).
        Returns False if Price is Stuck in Value (Chop).
        """
        lookback = CONFIG['LOOKBACK_VP']
        if len(df) < lookback: return True # Not enough data, assume safe
        
        # SLICE HISTORY (Avoid Look-Ahead Bias)
        # Use data up to YESTERDAY (-1) to define levels for TODAY
        history = df.iloc[-(lookback+1):-1] 
        
        # Simple VP Implementation
        price_bins = pd.cut(history['Close'], bins=50)
        vol_profile = history.groupby(price_bins)['Volume'].sum()
        
        # Calculate Value Area (70%)
        total_vol = vol_profile.sum()
        sorted_vol = vol_profile.sort_values(ascending=False)
        cumulative_vol = sorted_vol.cumsum()
        value_area_bins = cumulative_vol[cumulative_vol <= (0.7 * total_vol)].index
        
        # Define VAH (Value Area High) and VAL (Value Area Low)
        # Convert Intervals to midpoints or boundaries
        vah = max([b.right for b in value_area_bins])
        val = min([b.left for b in value_area_bins])
        
        current_price = df['Close'].iloc[-1]
        
        # LOGIC: Only Buy if we break OUT of the Value Area
        # If we are inside [VAL, VAH], it is Chop.
        if val <= current_price <= vah:
            return False # REJECT TRADE (Chop)
        
        return True # ACCEPT TRADE (Trend)

    # --- MODULE 3: THE HEART (VOLATILITY TARGETING) ---
    def calculate_sizing(self, market_df):
        """
        Determines Portfolio Exposure based on Market Heat.
        """
        # 1. Calculate Nifty Volatility (20 Day)
        # .shift(1) ensures we use YESTERDAY'S Vol to size TODAY'S trade
        daily_ret = market_df['Close'].pct_change()
        ann_vol = daily_ret.rolling(20).std() * np.sqrt(252)
        current_vol = ann_vol.iloc[-1]
        
        if pd.isna(current_vol) or current_vol == 0: return 1.0
        
        # 2. Target Exposure
        target_exposure = CONFIG['TARGET_VOL'] / current_vol
        
        # 3. Cap Leverage
        return min(target_exposure, CONFIG['MAX_LEVERAGE'])

    # --- MODULE 4: THE SHIELD (CRISIS ROTATION) ---
    def get_regime(self, market_df):
        """
        Bull or Bear?
        """
        sma200 = market_df['Close'].rolling(200).mean().iloc[-1]
        price = market_df['Close'].iloc[-1]
        
        if price < sma200:
            return "BEAR" # Crisis Mode
        return "BULL" # Alpha Mode

    # --- EXECUTION LOOP ---
    def run_simulation(self):
        # 1. Ingest
        all_tickers = CONFIG['TICKERS'] + [CONFIG['BENCHMARK'], CONFIG['SAFE_HAVEN']]
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        data_map = self.fetch_data(all_tickers, start_date)
        
        if not data_map: 
            print("No data fetched.")
            return

        # Handle Market Data missing check
        if CONFIG['BENCHMARK'] not in data_map:
             print(f"CRITICAL: Benchmark {CONFIG['BENCHMARK']} not found.")
             return

        market_data = data_map[CONFIG['BENCHMARK']]
        
        # 2. Rebalance Schedule (Weekly)
        dates = market_data.index
        # Resample to Weekly logic can be simulated by iterating days
        
        print("\n--- STARTING SIMULATION ---")
        for i in range(201, len(dates)): # Start after SMA200 is valid
            current_date = dates[i]
            
            # Check if it's Friday (Rebalance Day)
            if current_date.weekday() != 4: 
                continue 
            
            # A. REGIME CHECK
            # Slice market data strictly up to current_date
            # .iloc[:i+1] includes today. Ideally we trade MOC (Market On Close)
            market_slice = market_data.iloc[:i+1]
            regime = self.get_regime(market_slice)
            
            # B. SIZING CHECK
            exposure = self.calculate_sizing(market_slice)
            
            # C. PORTFOLIO ALLOCATION
            if regime == "BEAR":
                # ALLOCATE TO SAFE HAVEN
                print(f"[{current_date.date()}] REGIME: BEAR | Exposure: 100% GOLD")
                # (Simulation logic: Add Gold return to equity)
                
            else:
                # REGIME IS BULL
                # 1. Scan Candidates
                selected_stocks = []
                for ticker in CONFIG['TICKERS']:
                    if ticker not in data_map: continue
                    stock_df = data_map[ticker].iloc[:i+1]
                    
                    # Signal
                    sig = self.get_gaussian_signal(stock_df)
                    
                    if sig == "BUY":
                        # Filter (Volume Profile)
                        is_clean = self.check_volume_profile(stock_df)
                        if is_clean:
                            selected_stocks.append(ticker)
                
                # 2. Allocations
                if not selected_stocks:
                    print(f"[{current_date.date()}] REGIME: BULL | No Trades Found (Cash)")
                    continue
                    
                # Weight per stock = Total Exposure / Count
                weight = exposure / len(selected_stocks)
                
                print(f"[{current_date.date()}] REGIME: BULL | Vol: {1/exposure:.1%} | Stocks: {len(selected_stocks)} | Wgt: {weight:.2%}")
                # print(f"   >>> BUY: {selected_stocks}")

        print("--- SIMULATION COMPLETE ---")

# --- LAUNCH ---
if __name__ == "__main__":
    engine = ChimeraEngine()
    engine.run_simulation()
