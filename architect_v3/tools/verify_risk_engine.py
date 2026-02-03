import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.risk_engine import GarchEvtRiskEngine

def run_verification():
    print("--- QUANTUM RISK ENGINE: GARCH-EVT DIAGNOSTIC ---")
    
    # 1. Fetch Data
    ticker = "^NSEI"
    print(f"Fetching {ticker} history (10y)...")
    df = yf.download(ticker, period="10y", progress=False, auto_adjust=True)
    
    if df.empty:
        print("Error: No data fetched.")
        return

    # Handle MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs('Close', axis=1, level=0).iloc[:, 0]
    else:
        close = df['Close']
        
    returns = close.pct_change().dropna()
    print(f"Data Points: {len(returns)}")
    
    # 2. Run Engine
    engine = GarchEvtRiskEngine(confidence_level=0.99)
    print("Fitting GARCH(1,1) + EVT (Pareto Tails)...")
    
    try:
        var_limit, next_vol = engine.calculate_dynamic_var(returns)
        
        # 3. Results
        current_price = close.iloc[-1]
        stop_price = current_price * (1 + var_limit)
        
        print("\n--- RESULTS: DYNAMIC RISK LIMITS ---")
        print(f"Current Price:       {current_price:,.2f}")
        print(f"Forecast Volatility: {next_vol*100:.2f}% (Daily)")
        print(f"Forecast Volatility: {next_vol*np.sqrt(252)*100:.2f}% (Annualized)")
        print(f"VaR (99% Confidence):{var_limit*100:.2f}%")
        print(f"Dynamic Stop Loss:   {stop_price:,.2f} (Implied)")
        print(f"Static 3% Stop:      {current_price * 0.97:,.2f}")
        
        print("\n--- MODEL DIAGNOSTICS ---")
        print("Distribution: Skewed Student-t (GARCH)")
        print("Tail Physics: Generalized Pareto (EVT)")
        print("Note: If VaR > Static Stop, the market is 'Safe'. If VaR < Static Stop, Risk is Extreme.")
        
    except Exception as e:
        print(f"Model Execution Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_verification()
