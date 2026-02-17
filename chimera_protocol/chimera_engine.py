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
    'LOOKBACK_ENERGY': 21,
    
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

def validate_vacuum_with_depth(signal, order_book_data):
    """
    Verifies if the Vacuum is real using Top 5 Order Book.
    """
    if order_book_data is None:
        return True # Fallback if data missing (Dangerous but allows trade)

    friction = order_book_data.get('friction', 0.0)
    
    # RULE: Do not buy if Sellers are 3x stronger than Buyers
    if signal == "BULL_TURBO" and friction > 3.0:
        print(f"[BLOCK] TURBO BLOCKED: Wall Detected. Sell Pressure 3x > Buy Pressure (Friction: {friction:.2f}).")
        return False
        
    return True

class ChimeraEngineFinal:
    def __init__(self, override_params=None):
        print("--- CHIMERA PROTOCOL: INITIALIZING TRIANGULATION SENSORS ---")
        if override_params:
            print(f"🧬 [GENOME] Overriding Parameters: {override_params}")
            # Update CONFIG keys that match
            for k, v in override_params.items():
                if k in CONFIG:
                    CONFIG[k] = v
                # Also check uppercase keys mapping to DNA keys (which might be lowercase)
                k_upper = k.upper()
                if k_upper in CONFIG:
                    CONFIG[k_upper] = v
                    
        self.trade_log = [] 
        self.weekly_returns = [] # For Rolling Sharpe
        self.last_allocations = [] # [(ticker, weight, entry_price)] 
        
    def fetch_data(self, tickers, start_date):
        import polars as pl
        print(f"Fetching Data for {len(tickers)} assets from {start_date}...")
        
        start_dt = pd.to_datetime(start_date)
        clean_data = {}
        
        # 1. Identify Data Sources
        indices = list(CONFIG['INDICES'].values())
        safe_haven = [CONFIG['SAFE_HAVEN']]
        
        # Indices & Safe Haven usually via YFinance (unless mapped)
        remote_symbols = indices + safe_haven
        local_symbols = tickers 
        
        # --- A) LOCAL PARQUET FETCH (STOCKS) ---
        print(f"   [Source: LOCAL] Loading {len(local_symbols)} tickers from Data Lake...")
        lake_dir = "data/nse"
        
        for t in local_symbols:
            file_path = f"{lake_dir}/{t}.parquet"
            if os.path.exists(file_path):
                try:
                    # Read with Polars (Fast)
                    # Filter by date pushed down to scan if possible, or just filter after
                    pldf = pl.read_parquet(file_path)
                    
                    # Filter Date
                    pldf = pldf.filter(pl.col("timestamp") >= start_dt)
                    
                    if not pldf.is_empty():
                        # Convert to Pandas
                        pdf = pldf.to_pandas()
                        
                        # Standardize Columns (Polars is lowercase, Engine expects Title Case)
                        # Rename: timestamp -> Date, open -> Open, etc.
                        rename_map = {
                            "timestamp": "Date",
                            "open": "Open", "high": "High", "low": "Low", 
                            "close": "Close", "volume": "Volume"
                        }
                        pdf = pdf.rename(columns=rename_map)
                        
                        # Set Index
                        pdf.set_index("Date", inplace=True)
                        
                        # Ensure numeric
                        cols = ["Open", "High", "Low", "Close", "Volume"]
                        for c in cols:
                            if c in pdf.columns:
                                pdf[c] = pd.to_numeric(pdf[c])
                        
                        clean_data[t] = pdf
                except Exception as e:
                    print(f"[WARN] Corrupt/Read Error for {t}: {e}")
            else:
                print(f"[WARN] Missing Parquet for {t}. Skipping.")

        # --- B) REMOTE FETCH (INDICES) ---
        if remote_symbols:
            print(f"   [Source: REMOTE] Fetching {len(remote_symbols)} indices via YFinance...")
            try:
                # Use group_by='ticker' for reliable structure
                data = yf.download(remote_symbols, start=start_date, group_by='ticker', progress=False)
                
                if not data.empty:
                    for t in remote_symbols:
                        try:
                            # Handle MultiIndex or Single Level
                            if isinstance(data.columns, pd.MultiIndex):
                                if t in data.columns.levels[0]:
                                    df = data[t].copy()
                                    df = df.dropna(how='all')
                                    if not df.empty:
                                        clean_data[t] = df
                            else:
                                # Single ticker download returns single level columns
                                # But we passed a list, so yf usually returns multiindex 
                                # unless list has length 1. 
                                clean_data[t] = data 
                        except Exception as e:
                            pass
            except Exception as e:
                print(f"[ERROR] Remote Fetch Failed: {e}")

        print(f"DEBUG: Valid Data for {len(clean_data)} assets.")
        return clean_data

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
    def run_simulation(self, tickers=None, start_date=None):
        # 1. Ingest
        # User requested backtest from July 2021. 
        # Data starts ~Aug 2020. Warmup (250 days) requires setting start to June 2020.
        if start_date is None:
            start_date = "2020-06-01" 
        
        target_tickers = tickers if tickers else CONFIG['TICKERS']
        
        data_map = self.fetch_data(target_tickers, start_date)
        
        if not data_map: 
            print("No Data Data Loaded.")
            return

        # Use Local Data for Calendar (More Reliable/Fresh than Yahoo)
        calendar = None
        
        # Priority 1: Use a Liquid Stock from Local Lake (e.g. RELIANCE/TCS/HDFCBANK)
        # We check through the loaded data map for a high-quality ticker
        for preferred in ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY']:
            if preferred in data_map:
                calendar = data_map[preferred]
                print(f"Calendar Source: Local Lake ({preferred})")
                break
        
        # Priority 2: Use First Available Local Ticker
        if calendar is None and len(data_map) > 0:
             first_key = list(data_map.keys())[0]
             calendar = data_map[first_key]
             print(f"Calendar Source: {first_key} (Fallback)")

        # Priority 3: Remote Index (Least Reliable for Live Date)
        if calendar is None:
            calendar = data_map.get(CONFIG['INDICES']['BROAD'])

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

            # --- PERFORMANCE KILL-SWITCH (Audit Requirement 3) ---
            # 1. Calc Realized Return of LAST Week's bets
            weekly_pnl = 0.0
            if self.last_allocations:
                for t, w, entry_p in self.last_allocations:
                    try:
                        # Get current close
                        curr_p = data_map[t].loc[data_map[t].index <= current_date].iloc[-1]['Close']
                        # Simple return contribution
                        ret = ((curr_p - entry_p) / entry_p) * w
                        weekly_pnl += ret
                    except:
                        pass
                self.weekly_returns.append(weekly_pnl)
            
            # 2. Calc Rolling Sharpe (20 Weeks)
            perf_penalty = 1.0
            rolling_sharpe = 0.0
            if len(self.weekly_returns) >= 20:
                rh = np.array(self.weekly_returns[-20:])
                if np.std(rh) > 0:
                    rolling_sharpe = (np.mean(rh) / np.std(rh)) * np.sqrt(52)
            
            # 3. Kill Switch Logic
            if len(self.weekly_returns) > 20 and rolling_sharpe < -0.5:
                perf_penalty = 0.5
                reason += f" [KILL-SWITCH: Sharpe {rolling_sharpe:.2f}]"
            elif len(self.weekly_returns) > 20 and rolling_sharpe < 0.0:
                 perf_penalty = 0.75
                 reason += f" [CAUTION: Sharpe {rolling_sharpe:.2f}]"

            # B. VOLATILITY SIZING
            # Calc Vol on Broad Market
            # Use Calendar as proxy for Broad
            broad_slice = calendar.iloc[:i+1]
            returns = broad_slice['Close'].pct_change()
            current_vol = returns.tail(20).std() * np.sqrt(252)
            
            if current_vol == 0 or np.isnan(current_vol): current_vol = 0.15 # Fallback
            
            # Apply Performance Penalty to Base Leverage
            base_leverage = min(CONFIG['TARGET_VOL'] / current_vol, CONFIG['MAX_LEVERAGE']) * perf_penalty
            
            # C. DECISION TREE
            if regime == "BEAR":
                # LOG CRISIS ACTION
                # Log as one entry to represent the portfolio state
                self.trade_log.append({
                    'date': current_date, 'ticker': 'CASH', 'close': 1.0, 'weight': 1.0,
                    'kinetic_energy': 0.0, 'structure_tag': 'SAFE', 'leverage_mult': 0.0, 
                    'market_state': 'BEAR_DEFENSE', 'decision_reason': reason, 'nifty_vol': vix/100, 
                    'fwd_return': 0.0, 'net_pnl': 0.0, 'efficiency': 0.0
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
                    'kinetic_energy': 0.0, 'structure_tag': 'NONE', 'leverage_mult': 0.0, 
                    'market_state': 'BULL_IDLE', 'decision_reason': 'No Candidates', 'nifty_vol': vix/100,
                    'fwd_return': 0.0, 'net_pnl': 0.0, 'efficiency': 0.0
                })
                continue

            # E. ALLOCATION
            weight_per_stock = base_leverage / len(top_picks)
            final_weights = {}
            
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
                final_weights[ticker] = {'weight': final_weight, 'energy': energy, 'stock_slice': stock_slice, 'trade_note': trade_note, 'structure': structure, 'lev_mult': lev_mult}

            # 3. WEIGHT NORMALIZATION (Hard Cap - Audit Requirement 1)
            total_net_exposure = sum(abs(item['weight']) for item in final_weights.values()) # Use abs just in case
            max_allowed = CONFIG['MAX_LEVERAGE'] * perf_penalty # Double enforce metric
            
            if total_net_exposure > max_allowed:
                scale_factor = max_allowed / total_net_exposure
                for t in final_weights:
                    final_weights[t]['weight'] *= scale_factor
                    final_weights[t]['trade_note'] += " (Norm)"
            
            self.last_allocations = [] # Reset for next tracking
            
            # 4. EXECUTION
            for ticker, item in final_weights.items():
                final_weight = item['weight']
                energy = item['energy']
                stock_slice = item['stock_slice']
                trade_note = item['trade_note']
                structure = item['structure']
                lev_mult = item['lev_mult']
                
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
                        'structure_tag': structure,
                        
                        # --- GOVERNANCE TELEMETRY (For Reality Checks) ---
                        'meta_scale': perf_penalty,           # Architecture Health
                        'capital_scale': base_leverage / perf_penalty if perf_penalty > 0 else 0, # Pure Vol Targeting
                        'adaptive_scale': lev_mult,           # Market State Multiplier
                        'genome_scale': 1.0,                  # Parameter DNA (Static in backtest)
                        'family_scale': 1.0,                  # Alias for Reality Test
                        'drift_scale': 1.0                    # Model Drift (Placeholder)
                    })
                    
                    # Store for Next Week's Sharpe Calc
                    self.last_allocations.append((ticker, final_weight, curr_price))

        # EXPORT BLACK BOX
        log_df = pd.DataFrame(self.trade_log)
        filename = "data/chimera_blackbox_final.csv"
        os.makedirs("data", exist_ok=True)
        log_df.to_csv(filename, index=False)
        print(f"--- PROTOCOL COMPLETE. BLACK BOX SAVED: {filename} ---")
        return log_df

if __name__ == "__main__":
    eng = ChimeraEngineFinal()
    eng.run_simulation()
