import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

def run_volatility_targeting_simulation():
    print("ARCHITECT V10: VOLATILITY TARGETING (THE DIMMER SWITCH)")
    
    # 1. FETCH DATA
    # We need Nifty for the trend, and VIX for the 'Temperature'
    print("Fetching Nifty & VIX Data (15y)...")
    df = yf.download(['^NSEI', '^INDIAVIX'], period="15y", progress=False)['Close']
    
    # Handle MultiIndex logic
    # yfinance v0.2+ returns MultiIndex columns like (price, ticker)
    # We expect columns to be ^INDIAVIX and ^NSEI (or vice versa)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1) # Ticker is usually level 1? Or 0?
        # Let's check typical structure: ('Close', '^NSEI') -> level 1 is ticker
        # If it was flattened earlier: index is date, columns are tickers.
    
    # Check what columns we actually have
    print(f"Columns found: {df.columns.tolist()}")
    
    # Rename for clarity (Robust map)
    col_map = {}
    for c in df.columns:
        if 'NSEI' in c: col_map[c] = 'Nifty'
        if 'INDIAVIX' in c: col_map[c] = 'VIX'
    
    df = df.rename(columns=col_map)
    
    # Ensure usage columns exist
    if 'Nifty' not in df.columns or 'VIX' not in df.columns:
        print("CRITICAL: Could not identify Nifty or VIX columns.")
        return

    df = df.dropna()

    # 2. DEFINITIONS
    TARGET_VOL = 0.15  # We want our portfolio to move 15% a year (Institutional Standard)
    MAX_LEVERAGE = 1.5 # Allow 1.5x in calm times
    
    # 3. CALCULATE REALIZED VOLATILITY (The Speedometer)
    # We use a 20-day rolling window of Nifty volatility
    df['Nifty_Ret'] = df['Nifty'].pct_change()
    df['Realized_Vol_Ann'] = df['Nifty_Ret'].rolling(20).std() * np.sqrt(252)
    
    # 4. THE DIMMER SWITCH FORMULA
    # Allocation % = Target_Vol / Current_Vol
    # If Market is wild (30% Vol), we allocate 15/30 = 50% Exposure
    # If Market is calm (10% Vol), we allocate 15/10 = 150% Exposure
    
    df['Target_Exposure'] = (TARGET_VOL / df['Realized_Vol_Ann']).shift(1) # No look-ahead
    
    # Cap the leverage
    df['Target_Exposure'] = df['Target_Exposure'].clip(upper=MAX_LEVERAGE, lower=0.0)
    
    # 5. SIMULATE RETURNS
    # Strategy Return = Nifty_Return * Exposure
    # (Assuming we use your Gaussian picks, they roughly correlate to Nifty * Alpha. 
    # For this test, we assume your picks = Nifty Beta).
    
    df['Strat_Ret'] = df['Nifty_Ret'] * df['Target_Exposure']
    
    # 6. EQUITY CURVES
    df['Equity_BuyHold'] = (1 + df['Nifty_Ret']).cumprod() * 100
    df['Equity_VolTarget'] = (1 + df['Strat_Ret']).cumprod() * 100
    
    # 7. METRICS
    bh_dd = (df['Equity_BuyHold'] - df['Equity_BuyHold'].cummax()) / df['Equity_BuyHold'].cummax()
    vt_dd = (df['Equity_VolTarget'] - df['Equity_VolTarget'].cummax()) / df['Equity_VolTarget'].cummax()
    
    print(f"\n--- RESULTS ---")
    print(f"Buy & Hold Max DD:      {bh_dd.min():.2%}")
    print(f"Vol Targeted Max DD:    {vt_dd.min():.2%}")
    print(f"Final Equity (BH):      {df['Equity_BuyHold'].iloc[-1]:.0f}")
    print(f"Final Equity (Strat):   {df['Equity_VolTarget'].iloc[-1]:.0f}")
    
    # 8. VISUALIZATION
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Equity_BuyHold'], name='Nifty (Buy & Hold)', line=dict(color='grey')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Equity_VolTarget'], name='Vol Targeted (Smooth)', line=dict(color='#00ffcc', width=2)))
    
    # Show Exposure on secondary axis
    fig.add_trace(go.Scatter(x=df.index, y=df['Target_Exposure']*100, name='Exposure %', line=dict(color='orange', width=1), yaxis='y2'))
    
    fig.update_layout(
        title="ARCHITECT V10: The 'Dimmer Switch' Approach",
        template="plotly_dark",
        yaxis2=dict(title="Exposure %", overlaying='y', side='right', range=[0, 200])
    )
    
    fig.write_html("vol_target_audit.html")
    print("Report Generated: vol_target_audit.html")

if __name__ == "__main__":
    run_volatility_targeting_simulation()
