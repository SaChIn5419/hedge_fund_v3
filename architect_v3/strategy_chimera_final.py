import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import linregress
from datetime import datetime, timedelta
import os

# --- CONFIGURATION (THE CONTROL ROOM) ---
CONFIG = {
    'CAPITAL': 1000000,
    'TARGET_VOL': 0.15,
    'MAX_LEVERAGE': 1.5,
    'REBALANCE_FREQ': 'W-FRI',
    
    # PARAMETERS
    'LOOKBACK_GAUSSIAN': 140,
    'LOOKBACK_VP': 60,
    'LOOKBACK_ENERGY': 20,
    
    # THE SENSOR ARRAY (Triangulation)
    'INDICES': {
        'BROAD': '^NSEI',     # Nifty 50 (Anchor - More reliable data)
        'RISK': '^CNXSC',     # Nifty Smallcap 100 (The Canary)
        'FEAR': '^INDIAVIX'   # Volatility
    },
    
    # ASSET UNIVERSE (Dynamic Selection Candidates)
    'TICKERS': [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
        'TATAMOTORS.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'TITAN.NS',
        'SUNPHARMA.NS', 'JSWSTEEL.NS', 'NTPC.NS', 'POWERGRID.NS', 'ITC.NS',
        'ADANIENT.NS', 'KPITTECH.NS', 'ZOMATO.NS', 'HAL.NS', 'TRENT.NS',
        'BSE.NS', 'CDSL.NS', 'MCX.NS', 'ANGELONE.NS'
    ],
    
    'SAFE_HAVEN': 'GOLDBEES.NS'
}

class ChimeraEngineFinal:
    def __init__(self):
        print("--- CHIMERA PROTOCOL: INITIALIZING TRIANGULATION SENSORS ---")
        self.trade_log = [] 
        
    def fetch_data(self, tickers, start_date):
        print(f"Fetching Data for {len(tickers)} assets from {start_date}...")
        # Add Indices to the fetch list
        indices = list(CONFIG['INDICES'].values())
        all_symbols = tickers + indices + [CONFIG['SAFE_HAVEN']]
        
        try:
            # Use group_by='ticker' for reliable structure
            data = yf.download(all_symbols, start=start_date, group_by='ticker', progress=False)
            
            clean_data = {}
            # Verify if data is empty
            if data.empty:
                print("DEBUG: Download returned empty DataFrame.")
                return {}

            for t in all_symbols:
                try:
                    if t in data.columns.levels[0]: # Check top level
                        df = data[t].copy()
                        # Drop NA (partial data is okay, empty is not)
                        df = df.dropna(how='all') 
                        if not df.empty:
                            clean_data[t] = df
                except Exception as e:
                    # Fallback for single ticker (no levels)
                    if t == all_symbols[0] and len(all_symbols) == 1:
                         clean_data[t] = data
            
            print(f"DEBUG: Valid Data for {len(clean_data)} tickers.")
            return clean_data
        except Exception as e:
            print(f"[CRITICAL ERROR] Data Fetch Failed: {e}")
            return {}

    # --- PHYSICS MODULES ---
    def get_energy(self, series):
        """Calculates Slope (Kinetic Energy)"""
        if len(series) < CONFIG['LOOKBACK_ENERGY']: return 0.0
        y = series.tail(CONFIG['LOOKBACK_ENERGY']).values
        x = np.arange(len(y))
        slope, _, _, _, _ = linregress(x, y)
        if y.mean() == 0: return 0.0
        return (slope / y.mean()) * 100

    def check_structure(self, df):
        """Volume Profile: Returns VACUUM (True) or TRAPPED (False)"""
        lookback = CONFIG['LOOKBACK_VP']
        if len(df) < lookback: return "N/A"
        
        hist = df.iloc[-(lookback+1):-1] # No lookahead
        current = df['Close'].iloc[-1]
        
        mean = hist['Close'].mean()
        std = hist['Close'].std()
        if std == 0: return "TRAPPED"
        z = (current - mean) / std
        
        if abs(z) < 1.0: return "TRAPPED"
        return "VACUUM"

    def get_gaussian_signal(self, df):
        period = CONFIG['LOOKBACK_GAUSSIAN']
        if len(df) < period: return "NEUTRAL"
        mean = df['Close'].rolling(period).mean().iloc[-1]
        std = df['Close'].rolling(period).std().iloc[-1]
        price = df['Close'].iloc[-1]
        
        if price > mean + 2*std: return "BREAKOUT"
        elif price < mean - 2*std: return "BREAKDOWN"
        return "NEUTRAL"

    # --- THE TRIANGULATION SENSOR ---
    def get_composite_regime(self, data_map, current_date):
        """
        Checks 3 Indices. If ANY fail, we go Defensive.
        """
        # 1. BROAD MARKET (Nifty 500)
        broad = data_map.get(CONFIG['INDICES']['BROAD'])
        broad_status = "BULL"
        if broad is not None:
            slice_b = broad[broad.index <= current_date]
            if not slice_b.empty:
                sma200 = slice_b['Close'].rolling(200).mean().iloc[-1]
                if slice_b['Close'].iloc[-1] < sma200:
                    broad_status = "BEAR"

        # 2. RISK CANARY (Smallcap) - THE MOST CRITICAL
        risk = data_map.get(CONFIG['INDICES']['RISK'])
        risk_status = "BULL"
        if risk is not None:
            slice_r = risk[risk.index <= current_date]
            if not slice_r.empty:
                # Use faster SMA for canary (50 Day)
                sma50 = slice_r['Close'].rolling(50).mean().iloc[-1]
                if slice_r['Close'].iloc[-1] < sma50:
                    risk_status = "BEAR"
        else:
             # Fallback if Smallcap data missing (e.g. yfinance fail)
             # Don't fail the whole engine, just warn and rely on others
             pass

        # 3. FEAR SIREN (VIX)
        fear = data_map.get(CONFIG['INDICES']['FEAR'])
        fear_val = 15.0 # Default
        if fear is not None:
            slice_f = fear[fear.index <= current_date]
            if not slice_f.empty:
                fear_val = slice_f['Close'].iloc[-1]
        
        # LOGIC: VETO POWER
        # If Smallcaps are dying OR VIX is Panic -> RED
        if risk_status == "BEAR" or fear_val > 24.0:
            return "BEAR", fear_val, "Canary Died / VIX Spike"
        
        # If Broad market is dead -> RED
        if broad_status == "BEAR":
            return "BEAR", fear_val, "Broad Market Trend"

        return "BULL", fear_val, "All Systems Go"

    # --- EXECUTION LOOP ---
    def run_simulation(self):
        # 1. Ingest
        start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d') # 10 Years covers 2018
        data_map = self.fetch_data(CONFIG['TICKERS'], start_date)
        
        if not data_map: 
            print("No Data Data Loaded.")
            return

        # Use Broad index for calendar
        calendar = data_map.get(CONFIG['INDICES']['BROAD'])
        # Fallback to Safe Haven if Broad missing
        if calendar is None: 
            print("WARNING: BROAD Index missing. Trying Safe Haven.")
            calendar = data_map.get(CONFIG['SAFE_HAVEN'])
        
        # Fallback to First Available Ticker
        if calendar is None and len(data_map) > 0:
            print("WARNING: Safe Haven missing. Using first ticker.")
            calendar = list(data_map.values())[0]

        if calendar is None: 
            print("CRITICAL: No data available for calendar.")
            return
            
        print(f"Calendar Source loaded with {len(calendar)} rows.")
        dates = calendar.index
        print(f"\n--- SIMULATING {len(dates)} DAYS ---")
        
        for i in range(250, len(dates)):
            current_date = dates[i]
            
            # Weekly Rebalance (Friday) -> 4
            if current_date.weekday() != 4: continue 
            
            # A. TRIANGULATION CHECK
            regime, vix, reason = self.get_composite_regime(data_map, current_date)
            
            # B. VOLATILITY SIZING
            # Calc Vol on Broad Market
            # Use Calendar as proxy for Broad
            broad_slice = calendar.iloc[:i+1]
            returns = broad_slice['Close'].pct_change()
            current_vol = returns.tail(20).std() * np.sqrt(252)
            
            if current_vol == 0 or np.isnan(current_vol): current_vol = 0.15 # Fallback
            base_leverage = min(CONFIG['TARGET_VOL'] / current_vol, CONFIG['MAX_LEVERAGE'])
            
            # C. DECISION TREE
            if regime == "BEAR":
                # LOG CRISIS ACTION
                # Log as one entry to represent the portfolio state
                self.trade_log.append({
                    'date': current_date, 'ticker': 'CASH', 'close': 1.0, 'weight': 1.0,
                    'energy': 0, 'structure': 'SAFE', 'leverage_mult': 0.0, 
                    'market_state': 'BEAR_DEFENSE', 'decision_reason': reason, 'nifty_vol': vix/100, 
                    'fwd_return': 0.0, 'net_pnl': 0.0, 'kinetic_energy': 0.0, 'efficiency': 0.0
                })
                continue
            
            # D. BULL MODE: DYNAMIC SELECTION
            candidates = []
            for ticker in CONFIG['TICKERS']:
                if ticker not in data_map: continue
                stock_df = data_map[ticker]
                stock_slice = stock_df[stock_df.index <= current_date]
                
                if len(stock_slice) < 200: continue
                
                sig = self.get_gaussian_signal(stock_slice)
                if sig == "BREAKOUT" or sig == "NEUTRAL":
                    en = self.get_energy(stock_slice['Close'])
                    candidates.append((ticker, en, stock_slice))
            
            # Sort by Energy (Momentum) -> Pick Top 5
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_picks = candidates[:5]
            
            if not top_picks:
                self.trade_log.append({
                    'date': current_date, 'ticker': 'CASH', 'close': 1.0, 'weight': 1.0,
                    'energy': 0, 'structure': 'NONE', 'leverage_mult': 0.0, 
                    'market_state': 'BULL_IDLE', 'decision_reason': 'No Candidates', 'nifty_vol': vix/100,
                    'fwd_return': 0.0, 'net_pnl': 0.0, 'kinetic_energy': 0.0, 'efficiency': 0.0
                })
                continue

            # E. ALLOCATION
            weight_per_stock = base_leverage / len(top_picks)
            
            for ticker, energy, stock_slice in top_picks:
                structure = self.check_structure(stock_slice)
                
                # Dynamic Leverage Multiplier
                lev_mult = 1.0
                trade_note = "Standard Entry" # Default
                
                if vix > 24: # Redundant check, but safe for 'soft' panic
                    lev_mult = 0.5
                    trade_note = "VIX Penalty"
                elif energy > 0.25 and structure == "VACUUM":
                    lev_mult = 1.5
                    trade_note = "Turbo (High Energy)"
                elif structure == "TRAPPED":
                    lev_mult = 0.0 
                    trade_note = "Volume Profile (Chop)"
                
                final_weight = weight_per_stock * lev_mult
                
                if final_weight > 0:
                    # Calculate Fwd Return for Log (Simulated)
                    # Use next week return? Or next day? The dashboard expects net_pnl
                    # Let's approx 5-day fwd return since we rebalance weekly
                    curr_price = stock_slice['Close'].iloc[-1]
                    # Peek ahead
                    try:
                        future_slice = data_map[ticker][data_map[ticker].index > current_date]
                        if not future_slice.empty:
                            # 5 days later approx
                            if len(future_slice) >= 5:
                                next_price = future_slice['Close'].iloc[4]
                            else:
                                next_price = future_slice['Close'].iloc[-1]
                            fwd_ret = (next_price - curr_price) / curr_price
                        else:
                            fwd_ret = 0.0
                    except:
                        fwd_ret = 0.0

                    self.trade_log.append({
                        'date': current_date,
                        'ticker': ticker,
                        'close': curr_price,
                        'weight': final_weight,
                        'kinetic_energy': energy/100, # Normalize for dashboard
                        'efficiency': 1.0 if structure=="VACUUM" else 0.0,
                        'leverage_mult': lev_mult,
                        'market_state': 'BULL_TURBO' if regime=="BULL" else 'BEAR_DEFENSE',
                        'decision_reason': trade_note,
                        'nifty_vol': vix/100,
                        'fwd_return': fwd_ret,
                        'net_pnl': fwd_ret * 100000 * final_weight, # Fake PnL scaling
                        'structure_tag': structure
                    })

        # EXPORT BLACK BOX
        log_df = pd.DataFrame(self.trade_log)
        filename = "chimera_blackbox_final.csv"
        log_df.to_csv(filename, index=False)
        print(f"--- PROTOCOL COMPLETE. BLACK BOX SAVED: {filename} ---")

if __name__ == "__main__":
    eng = ChimeraEngineFinal()
    eng.run_simulation()
