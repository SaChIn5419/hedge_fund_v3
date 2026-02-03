import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import linregress
from datetime import datetime, timedelta

# --- CONFIGURATION ---
CONFIG = {
    'CAPITAL': 1000000,
    'REBALANCE_FREQ': 'W-FRI',
    
    # PARAMETERS
    'LOOKBACK_GAUSSIAN': 140,
    'LOOKBACK_VP': 60,
    'LOOKBACK_ENERGY': 20,
    
    # SENSORS
    'INDICES': {
        'BROAD': '^CRSLDX',   # Nifty 500
        'RISK': '^CNXSC',     # Smallcap 100
        'FEAR': '^INDIAVIX'   # VIX
    },
    
    # UNIVERSE (Dynamic Selection)
    'TICKERS': [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
        'TATAMOTORS.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'TITAN.NS',
        'SUNPHARMA.NS', 'JSWSTEEL.NS', 'NTPC.NS', 'POWERGRID.NS', 'ITC.NS',
        'ADANIENT.NS', 'KPITTECH.NS', 'ZOMATO.NS', 'HAL.NS', 'TRENT.NS',
        'BSE.NS', 'CDSL.NS', 'MCX.NS', 'ANGELONE.NS', 'BEL.NS', 'COALINDIA.NS'
    ],
    
    'SAFE_HAVEN': 'GOLDBEES.NS'
}

class ChimeraEngineUnlevered:
    def __init__(self):
        print("--- CHIMERA PROTOCOL: UNLEVERED BLACK BOX ---")
        self.trade_log = [] 
        
    def fetch_data(self, tickers, start_date):
        print(f"Fetching Data for {len(tickers)} assets from {start_date}...")
        indices = list(CONFIG['INDICES'].values())
        all_symbols = tickers + indices + [CONFIG['SAFE_HAVEN']]
        
        try:
            data = yf.download(all_symbols, start=start_date, progress=False)
            
            # Robust Flattening for yfinance v0.2+
            if isinstance(data.columns, pd.MultiIndex):
                # Check levels
                # If Price is level 0 and Ticker is level 1
                if 'Ticker' in data.columns.names:
                    ticker_level = data.columns.names.index('Ticker')
                else:
                    ticker_level = 1 # Assumption
                
                print(f"DEBUG DATA SHAPE: {data.shape}")
                print(f"DEBUG COLS: {data.columns.names}")
                
                try:
                    # Stack ticker to index
                    data_stacked = data.stack(level=ticker_level, future_stack=True)
                    data_stacked.rename_axis(['Date', 'Ticker'], inplace=True)
                    data_stacked = data_stacked.reset_index(level=1)
                except Exception as e:
                    print(f"Stacking failed: {e}. Trying alternate method.")
                    # Alternate: Loop columns
                    clean_data = {}
                    # We expect columns like (Close, Ticker) or (Ticker, Close)
                    # Let's just iterate known symbols
                    for t in all_symbols:
                        try:
                            # Try to extract cross section for this ticker
                            # This is tricky without knowing exact structure
                            # Let's rely on standard yf structure: Price, Ticker
                            df = data.xs(t, axis=1, level=ticker_level, drop_level=True)
                             # Force numeric
                            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                            for c in cols: 
                                if c in df.columns: 
                                    df[c] = pd.to_numeric(df[c], errors='coerce')
                            clean_data[t] = df.dropna()
                        except:
                            pass
                    return clean_data
                
                clean_data = {}
                for t in all_symbols:
                    df = data_stacked[data_stacked['Ticker'] == t].copy()
                    # Force numeric conversion
                    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for c in cols:
                        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
                    
                    if not df.empty:
                        # Ensure sorted index and drop only if Close is NaN
                        clean_data[t] = df.sort_index().dropna(subset=['Close'])
                
                print(f"DEBUG: Found data for {list(clean_data.keys())}")
                return clean_data
            
            # If not MultiIndex (single ticker?)
            else:
                # Should not happen with multiple tickers request, but handle it
                print("Single index returned (unexpected for list).")
                return {}

        except Exception as e:
            print(f"[CRITICAL ERROR] Data Fetch Failed: {e}")
            return {}

    # --- PHYSICS MODULES ---
    def get_energy(self, series):
        if len(series) < CONFIG['LOOKBACK_ENERGY']: return 0.0
        y = series.tail(CONFIG['LOOKBACK_ENERGY']).values
        x = np.arange(len(y))
        slope, _, _, _, _ = linregress(x, y)
        return (slope / y.mean()) * 100

    def check_structure(self, df):
        lookback = CONFIG['LOOKBACK_VP']
        if len(df) < lookback: return "N/A"
        hist = df.iloc[-(lookback+1):-1]
        current = df['Close'].iloc[-1]
        
        z = (current - hist['Close'].mean()) / hist['Close'].std()
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

    # --- TRIANGULATION (BINARY REGIME) ---
    def check_market_health(self, data_map, current_date):
        """
        Returns (True/False, Reason, VIX_Value)
        """
        vix_val = 0.0
        
        # 1. Fear Gauge (VIX) - Check this first
        fear = data_map.get(CONFIG['INDICES']['FEAR'])
        if fear is not None:
            slice_f = fear[fear.index <= current_date]
            if not slice_f.empty:
                vix_val = slice_f['Close'].iloc[-1]
                if vix_val > 24.0:
                    return False, f"VIX Panic ({vix_val:.1f})", vix_val

        # 2. Risk Canary (Smallcap)
        risk = data_map.get(CONFIG['INDICES']['RISK'])
        if risk is not None:
            slice_r = risk[risk.index <= current_date]
            if not slice_r.empty:
                sma50 = slice_r['Close'].rolling(50).mean().iloc[-1]
                if slice_r['Close'].iloc[-1] < sma50:
                    return False, "Smallcap Breakdown", vix_val

        # 3. Broad Market Trend
        broad = data_map.get(CONFIG['INDICES']['BROAD'])
        if broad is not None:
            slice_b = broad[broad.index <= current_date]
            if not slice_b.empty:
                sma200 = slice_b['Close'].rolling(200).mean().iloc[-1]
                if slice_b['Close'].iloc[-1] < sma200:
                    return False, "Broad Market Downtrend", vix_val
        
        return True, "Safe", vix_val

    # --- SIMULATION LOOP ---
    def run_simulation(self):
        start_date = (datetime.now() - timedelta(days=365*7)).strftime('%Y-%m-%d')
        data_map = self.fetch_data(CONFIG['TICKERS'], start_date)
        
        if not data_map: 
            print("No data found.")
            return

        calendar = data_map.get(CONFIG['INDICES']['BROAD'])
        if calendar is None:
            print(f"Benchmark {CONFIG['INDICES']['BROAD']} data missing/None in data_map. Keys: {list(data_map.keys())}")
            return
        
        print(f"DEBUG: Benchmark DF Shape: {calendar.shape}")
        print(f"DEBUG: Benchmark Index: {calendar.index}")
            
        dates = calendar.index
        print(f"\n--- SIMULATING BINARY STRATEGY ({len(dates)} DAYS) ---")
        
        # We start loop later to allow indicators to warm up
        for i in range(250, len(dates)-5): # -5 to allow Forward Return calc
            current_date = dates[i]
            
            # Weekly Rebalance (Friday)
            if current_date.weekday() != 4: continue 
            
            # A. REGIME CHECK (The Kill Switch)
            is_safe, reason, vix = self.check_market_health(data_map, current_date)
            
            # B. CALCULATE FUTURE DATE (For Return Attribution)
            # We look approx 1 week ahead (5 trading days)
            future_idx = i + 5
            if future_idx >= len(dates): break
            future_date = dates[future_idx]
            
            if not is_safe:
                # GO TO CASH / GOLD
                # Calculate return of Safe Haven
                fwd_ret = 0.0
                safe_haven = data_map.get(CONFIG['SAFE_HAVEN'])
                if safe_haven is not None:
                    # Check if date exists in safe haven data
                    if current_date in safe_haven.index and future_date in safe_haven.index:
                        p1 = safe_haven.loc[current_date]['Close']
                        p2 = safe_haven.loc[future_date]['Close']
                        fwd_ret = (p2 / p1) - 1
                
                self.trade_log.append({
                    'date': current_date, 'ticker': 'SAFE_HAVEN', 
                    'close': 1.0, 'weight': 1.0,
                    'energy': 0, 'structure': 'SAFE', 'regime': 'BEAR', 
                    'reason': reason, 'benchmark_vix': vix,
                    'fwd_return': fwd_ret # CRITICAL FOR MONTE CARLO
                })
                continue
                
            # C. BULL MODE: PICK TOP 5
            candidates = []
            for ticker in CONFIG['TICKERS']:
                if ticker not in data_map: continue
                stock_df = data_map[ticker]
                # Slice strictly up to current date (No Lookahead for decision)
                stock_slice = stock_df[stock_df.index <= current_date]
                
                if len(stock_slice) < 200: continue
                
                sig = self.get_gaussian_signal(stock_slice)
                if sig == "BREAKOUT" or sig == "NEUTRAL":
                    # EXTRA BINARY FILTER: Is individual stock in chop?
                    structure = self.check_structure(stock_slice)
                    if structure == "VACUUM":
                        en = self.get_energy(stock_slice['Close'])
                        candidates.append((ticker, en, stock_slice))
            
            # Sort by Energy
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_picks = candidates[:5] # Max 5 stocks
            
            if not top_picks:
                self.trade_log.append({
                    'date': current_date, 'ticker': 'CASH', 
                    'close': 1.0, 'weight': 1.0,
                    'energy': 0, 'structure': 'NONE', 'regime': 'BULL', 
                    'reason': 'No Valid Setups', 'benchmark_vix': vix,
                    'fwd_return': 0.0
                })
                continue
                
            # D. ALLOCATION (Equal Weight)
            weight_per_stock = 1.0 / len(top_picks)
            
            for ticker, energy, stock_slice in top_picks:
                # Calculate Forward Return for this specific stock
                # We need the FUTURE price from the FULL dataframe (stock_df from data_map)
                # Note: 'stock_slice' is cut off at current_date. 
                # We access the original 'data_map[ticker]' to peek into next week.
                
                stock_full = data_map[ticker]
                current_price = stock_slice['Close'].iloc[-1]
                
                # Find price at future_date (or closest)
                # Safe logic: lookup index
                fwd_ret = 0.0
                try:
                    # We use searchsorted to find closest index in future
                    # But simpler is to rely on our loop index 'i' aligned with calendar
                    # Assumption: Ticker has data on future_date
                    if future_date in stock_full.index:
                        future_price = stock_full.loc[future_date]['Close']
                        fwd_ret = (future_price / current_price) - 1
                    else:
                        # Fallback: Find next available after current
                        next_avail = stock_full[stock_full.index > current_date].head(5)
                        if not next_avail.empty:
                            # Take the price ~5 days out
                            future_price = next_avail['Close'].iloc[-1] 
                            fwd_ret = (future_price / current_price) - 1
                except:
                    fwd_ret = 0.0

                self.trade_log.append({
                    'date': current_date,
                    'ticker': ticker,
                    'close': current_price,
                    'weight': weight_per_stock,
                    'energy': energy,
                    'structure': "VACUUM",
                    'regime': 'BULL',
                    'reason': "High Momentum Entry",
                    'benchmark_vix': vix,
                    'fwd_return': fwd_ret # CRITICAL FOR MONTE CARLO
                })

        # EXPORT
        log_df = pd.DataFrame(self.trade_log)
        filename = "chimera_blackbox_final.csv" # Overwrite for the Dashboard/MC to use
        log_df.to_csv(filename, index=False)
        print(f"--- BINARY SIMULATION COMPLETE. BLACK BOX SAVED: {filename} ---")

if __name__ == "__main__":
    eng = ChimeraEngineUnlevered()
    eng.run_simulation()
