# strategies/architect/main_executor.py
import time
import math
import pandas as pd
import os
try:
    from . import config
    from .alpha_engine import SignalProcessor
    from .regime_engine import RegimeDetector
except ImportError:
    import config
    from alpha_engine import SignalProcessor
    from regime_engine import RegimeDetector

print("ARCHITECT Online. Authenticating with Dhan...")
# dhan = dhanhq(config.DHAN_CLIENT_ID, config.DHAN_ACCESS_TOKEN) # Commented out for Local Mode
regime_engine = RegimeDetector()

def fetch_daily_data(symbol):
    """Fetches Daily Data for Indicator Calculation from Local Parquet"""
    # handle numeric/string mapping if needed. 
    # For now, we assume 'symbol' is the ticker name (e.g. 'TATAMOTORS.NS')
    
    # Map '13' to Nifty (Config) if needed
    if symbol == "13": symbol = config.NIFTY_TICKER
    
    # Logic: if starts with ^, use as is. Else ensure it has .NS (unless it already does)
    if not symbol.endswith(".NS") and not symbol.startswith("^"):
        symbol = f"{symbol}.NS"
        
    file_path = f"data/raw/{symbol}.parquet"
    
    if not os.path.exists(file_path):
        # Fallback for indices saving without .NS sometimes
        file_path_alt = f"data/raw/{symbol.replace('.NS','')}.parquet"
        if os.path.exists(file_path_alt):
            file_path = file_path_alt
        else:
            print(f"[ERROR] Data for {symbol} not found locally.")
            return pd.DataFrame()
    
    df = pd.read_parquet(file_path)
    df = df.rename(columns=str.lower)
    return df

def calculate_shares_to_buy(price):
    """Allocates exactly 1 Lakh per position"""
    return math.floor(config.CAPITAL_PER_POSITION / price)

def run_daily_cycle():
    print("Executing Daily Cycle...")
    
    if not os.path.exists("prime_assets.csv"):
        print("PRIME ASSETS NOT FOUND. Run screener_engine.py first.")
        return

    # 1. Load Prime Assets
    prime_assets = pd.read_csv("prime_assets.csv")['Ticker'].tolist()
    print(f"Loaded {len(prime_assets)} Prime Assets.")
    
    # 2. Check Macro Regime (Using Nifty 50)
    nifty_df = fetch_daily_data("13") 
    market_regime = regime_engine.fit_predict(nifty_df)
    
    print(f"MARKET REGIME: {market_regime}")
    
    if market_regime == 2:
        print("REGIME 2 DETECTED (CRASH). SYSTEM ABORT. NO TRADES TODAY.")
        return

    # 3. Analyze and Execute on the Top 5
    for ticker in prime_assets:
        try:
            # Dhan requires numeric IDs, so you need a mapper from Ticker -> ID
            # symbol_id = ticker 

            df = fetch_daily_data(ticker)
            if df.empty: continue
            
            latest_close = df['close'].iloc[-1]
            
            df = SignalProcessor.generate_signals(df)
            
            # Law 3: Zero Trust. We check the signal from YESTERDAY (Completed Candle). 
            # We buy TODAY at Market Open.
            if len(df) < 2: continue
            
            yesterday_signal = df['Bull_Signal'].iloc[-2]
            today_signal = df['Bull_Signal'].iloc[-1]
            
            # print(f"{ticker}: Signal T-1: {yesterday_signal}")
            
            if yesterday_signal == 1:
                qty = calculate_shares_to_buy(latest_close)
                print(f"SIGNAL: {ticker} | BUYING {qty} SHARES (CASH EQUITY) @ {latest_close}")
                # dhan.place_order(symbol_id, qty, 'BUY', 'MARKET', 'CNC')
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

if __name__ == "__main__":
    # In production, schedule this script to run at 09:15 AM Mon-Fri using cron/Task Scheduler
    run_daily_cycle()
