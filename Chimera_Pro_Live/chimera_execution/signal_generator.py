import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import linregress
from datetime import datetime
from .config import *

class ChimeraSignalGenerator:
    def __init__(self):
        print("--- CHIMERA SIGNAL GENERATOR INITIALIZED ---")
    
    def fetch_data(self, lookback_days=365):
        """
        Fetches 'Live' data for research/signal generation.
        Uses yfinance to match the logic of the original engine.
        """
        start_date = (datetime.now() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        print(f"📡 Fetching data for {len(TICKERS)} assets + Indices since {start_date}...")
        
        indices = list(INDICES.values())
        all_symbols = TICKERS + indices + [SAFE_HAVEN]
        
        try:
            # Group by ticker to handle multi-index structure correctly
            data = yf.download(all_symbols, start=start_date, group_by='ticker', progress=False)
            
            clean_data = {}
            if data.empty:
                print("❌ Data fetch returned empty.")
                return {}

            for t in all_symbols:
                try:
                    if t in data.columns.levels[0]:
                        df = data[t].copy()
                        df = df.dropna(how='all')
                        if not df.empty:
                            clean_data[t] = df
                except Exception:
                    # Fallback for single ticker fetch logic if structure differs
                    if t == all_symbols[0] and len(all_symbols) == 1:
                         clean_data[t] = data
            
            print(f"✅ Data fetched successfully for {len(clean_data)} assets.")
            return clean_data
        except Exception as e:
            print(f"❌ Critical Data Fetch Error: {e}")
            return {}

    # --- PHYSICS MODULES (PORTED) ---
    def get_energy(self, series):
        """Calculates Slope (Kinetic Energy)"""
        if len(series) < LOOKBACK_ENERGY: return 0.0
        y = series.tail(LOOKBACK_ENERGY).values
        x = np.arange(len(y))
        slope, _, _, _, _ = linregress(x, y)
        if y.mean() == 0: return 0.0
        return (slope / y.mean()) * 100

    def check_structure(self, df):
        """Volume Profile: Returns VACUUM (True) or TRAPPED (False)"""
        if len(df) < LOOKBACK_VP: return "N/A"
        
        hist = df.iloc[-(LOOKBACK_VP+1):-1] 
        current = df['Close'].iloc[-1]
        
        mean = hist['Close'].mean()
        std = hist['Close'].std()
        if std == 0: return "TRAPPED"
        z = (current - mean) / std
        
        if abs(z) < 1.0: return "TRAPPED"
        return "VACUUM"

    def get_gaussian_signal(self, df):
        if len(df) < LOOKBACK_GAUSSIAN: return "NEUTRAL"
        mean = df['Close'].rolling(LOOKBACK_GAUSSIAN).mean().iloc[-1]
        std = df['Close'].rolling(LOOKBACK_GAUSSIAN).std().iloc[-1]
        price = df['Close'].iloc[-1]
        
        if price > mean + 2*std: return "BREAKOUT"
        elif price < mean - 2*std: return "BREAKDOWN"
        return "NEUTRAL"

    # --- REGIME DETECTION ---
    def get_composite_regime(self, data_map):
        """
        Checks 3 Indices. If ANY fail, we go Defensive.
        """
        # 1. BROAD MARKET (Nifty 50)
        broad = data_map.get(INDICES['BROAD'])
        broad_status = "BULL"
        if broad is not None and not broad.empty:
            sma200 = broad['Close'].rolling(200).mean().iloc[-1]
            if broad['Close'].iloc[-1] < sma200:
                broad_status = "BEAR"

        # 2. RISK CANARY (Smallcap)
        risk = data_map.get(INDICES['RISK'])
        risk_status = "BULL"
        if risk is not None and not risk.empty:
            sma50 = risk['Close'].rolling(50).mean().iloc[-1]
            if risk['Close'].iloc[-1] < sma50:
                risk_status = "BEAR"

        # 3. FEAR SIREN (VIX)
        fear = data_map.get(INDICES['FEAR'])
        fear_val = 15.0 # Default
        if fear is not None and not fear.empty:
            fear_val = fear['Close'].iloc[-1]
        
        # LOGIC: VETO POWER
        if risk_status == "BEAR" or fear_val > 24.0:
            return "BEAR", fear_val, "Canary Died / VIX Spike"
        
        if broad_status == "BEAR":
            return "BEAR", fear_val, "Broad Market Trend"

        return "BULL", fear_val, "All Systems Go"

    def generate_signals(self):
        """
        Main runner to generate TODAY's signals.
        Returns a DataFrame of signals ready for execution.
        """
        data_map = self.fetch_data()
        if not data_map: return pd.DataFrame()

        # 1. Regime Check
        regime, vix, reason = self.get_composite_regime(data_map)
        print(f"🚦 Regime: {regime} | VIX: {vix:.2f} | Reason: {reason}")

        if regime == "BEAR":
            print("🛡️ Defensive Mode Active. Allocating to CASH/SAFE HAVEN.")
            return pd.DataFrame([{
                'tradingsymbol': 'CASH', # Or Safe Haven Ticker
                'symboltoken': '',
                'weight': 0.0, # Or 1.0 for Safe Haven ETF
                'volatility': vix/100,
                'action': 'DEFENSE'
            }])

        # 2. Candidate Filtering
        candidates = []
        for ticker in TICKERS:
            if ticker not in data_map: continue
            stock_df = data_map[ticker]
            
            # Require minimum history
            if len(stock_df) < 200: continue
            
            sig = self.get_gaussian_signal(stock_df)
            if sig == "BREAKOUT" or sig == "NEUTRAL":
                en = self.get_energy(stock_df['Close'])
                candidates.append((ticker, en, stock_df))
        
        # 3. Selection (Top 5 by Energy)
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_picks = candidates[:5]
        
        if not top_picks:
            print("⚠️ Bull Regime but No Valid Candidates Found.")
            return pd.DataFrame()

        # 4. Volatility Sizing
        broad_idx = data_map.get(INDICES['BROAD'])
        current_vol = 0.15
        if broad_idx is not None:
            returns = broad_idx['Close'].pct_change()
            current_vol = returns.tail(20).std() * np.sqrt(252)
            if np.isnan(current_vol) or current_vol == 0: current_vol = 0.15
        
        base_leverage = min(TARGET_VOL / current_vol, MAX_LEVERAGE)
        weight_per_stock = base_leverage / len(top_picks)
        
        final_signals = []
        
        for ticker, energy, stock_df in top_picks:
            structure = self.check_structure(stock_df)
            
            # Dynamic Multiplier logic
            lev_mult = 1.0
            if vix > 24: lev_mult = 0.5
            elif energy > 0.25 and structure == "VACUUM": lev_mult = 1.5
            elif structure == "TRAPPED": lev_mult = 0.0 # Filter out chopped stocks
            
            final_weight = weight_per_stock * lev_mult
            
            # Only trade if weight is significant
            if final_weight > 0.01:
                final_signals.append({
                    'tradingsymbol': ticker, # Ensure this matches Broker Symbol
                    'symboltoken': '',       # Will need mapping later
                    'weight': final_weight,
                    'volatility': current_vol, # Market Vol for Scaling
                    'energy': energy,
                    'structure': structure
                })
                
        return pd.DataFrame(final_signals)
