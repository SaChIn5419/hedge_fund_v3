import pandas as pd
import yfinance as yf
import numpy as np
import os
import sys

# Add parent path to import config if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def load_tickers():
    # Try to load from active trade log first
    log_path = os.path.join("data", "chimera_blackbox_final.csv")
    tickers = []
    
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        if 'ticker' in df.columns:
            tickers = df['ticker'].unique().tolist()
            # Remove CASH if present
            if 'CASH' in tickers: tickers.remove('CASH')
    
    # Fallback to a default list if log empty
    if not tickers:
        print("No active tickers found in log. Using detailed default list.")
        tickers = [
            'RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'ICICIBANK.NS', 'INFY.NS',
            'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'LICI.NS', 'LT.NS'
        ]
        
    print(f"Analyzing {len(tickers)} tickers: {tickers}")
    return tickers

def calculate_friction(df):
    """
    Friction = Volume / DataRange
    High Friction = Huge Volume with No Price Movement (Absorption/Wall)
    """
    # Avoid zero division
    price_range = (df['High'] - df['Low']).replace(0, 0.05) 
    
    # Raw Friction
    df['Friction_Raw'] = df['Volume'] / price_range
    
    # Normalize (Z-Score) to find anomalies relative to that stock's history
    df['Friction_Z'] = (df['Friction_Raw'] - df['Friction_Raw'].mean()) / df['Friction_Raw'].std()
    
    return df

def run_analysis():
    print("--- CHIMERA: HISTORICAL FRICTION ANALYZER (90 DAYS) ---")
    tickers = load_tickers()
    
    all_anomalies = []
    
    for ticker in tickers:
        print(f"Scanning {ticker}...")
        try:
            # Fetch 90 days of hourly data to see intraday walls
            df = yf.download(ticker, period="3mo", interval="1h", progress=False)
            
            if df.empty:
                print(f"⚠️ No data for {ticker}")
                continue
                
            # Check structure (multi-index handling for yf)
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten
                df = df.xs(ticker, axis=1, level=1) if ticker in df.columns.levels[0] else df
                # If that failed and we still have multiindex, valid columns might be under the ticker name
                if isinstance(df.columns, pd.MultiIndex):
                     df.columns = df.columns.get_level_values(0)

            # Ensure we have required columns
            req = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in req):
                print(f"⚠️ Missing columns for {ticker}")
                continue
                
            df = calculate_friction(df)
            
            # Identify Walls: Friction Z-Score > 2.0 (Top 2.5% events)
            walls = df[df['Friction_Z'] > 3.0].copy()
            
            if not walls.empty:
                walls['ticker'] = ticker
                # Reset index to get datetime column
                walls = walls.reset_index() 
                # Rename 'index' or 'Date' or 'Datetime' to 'timestamp'
                date_col = [c for c in walls.columns if 'date' in str(c).lower() or 'time' in str(c).lower()][0]
                walls.rename(columns={date_col: 'timestamp'}, inplace=True)
                
                # Select relevant cols
                cols = ['timestamp', 'ticker', 'Close', 'Volume', 'Friction_Raw', 'Friction_Z']
                all_anomalies.append(walls[cols])
                
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {e}")
            
    if all_anomalies:
        final_df = pd.concat(all_anomalies)
        final_df = final_df.sort_values(['ticker', 'Friction_Z'], ascending=[True, False])
        
        out_path = os.path.join("data", "chimera_friction_analysis.csv")
        os.makedirs("data", exist_ok=True)
        final_df.to_csv(out_path, index=False)
        
        print(f"\n[DONE] ANALYSIS COMPLETE.")
        print(f"Found {len(final_df)} 'Hidden Wall' events.")
        print(f"Saved to: {out_path}")
        print("\nTop 5 Highest Friction Events (Likely Iceberg Orders):")
        print(final_df.head(5).to_string(index=False))
        
    else:
        print("\n✅ Analysis Complete. No extreme anomalies found.")

if __name__ == "__main__":
    run_analysis()
