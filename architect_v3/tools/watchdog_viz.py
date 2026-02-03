import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
BENCHMARK_TICKER = '^NSEI' # Nifty 50
TIMEFRAME = "10y"

def fetch_clean_data(ticker, period):
    """
    ARCHITECT PROTOCOL: ROBUST INGESTION
    Handles the yfinance Multi-Index/Single-Index ambiguity.
    """
    print(f"WATCHDOG VIZ: Fetching {ticker} history ({period})...")
    
    # 1. Download Data
    df = yf.download(ticker, period=period, progress=False)
    
    # 2. THE FIX: Flatten Multi-Index Columns
    # If columns look like ('Close', '^NSEI'), flatten to 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 3. Validation
    if df.empty:
        print(f"[CRITICAL] No data found for {ticker}. Check ticker symbol.")
        return pd.DataFrame()
        
    # 4. Standardize Index
    df.index = pd.to_datetime(df.index)
    
    return df

def run_diagnosis():
    # 1. Fetch Data with the Patch
    df = fetch_clean_data(BENCHMARK_TICKER, TIMEFRAME)
    if df.empty: return

    # 2. Physics Calculations (Now Safe)
    try:
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['Drawdown'] = (df['Close'] - df['Close'].cummax()) / df['Close'].cummax()
    except KeyError as e:
        print(f"[ERROR] Data Structure Mismatch: {e}")
        print("Columns Available:", df.columns.tolist())
        return

    # 3. Simulate Watchdog (Regime Guard)
    # Logic: If Price < SMA200, we are 'DEFENSIVE' (Cash/Gold).
    # We assume 'Defensive' asset holds value (0% return) or Gold (Inverse).
    # For visualization, we assume Cash (Flat line) during crashes.
    
    df['Strategy_Equity'] = 100.0 # Start at 100
    df['System_State'] = 1.0 # 1 = ON, 0 = OFF (Defensive)
    
    equity = [100.0]
    states = [1.0]
    
    prices = df['Close'].values
    smas = df['SMA_200'].values
    
    for i in range(1, len(df)):
        # Check Regime (Yesterday's Close vs SMA)
        prev_price = prices[i-1]
        prev_sma = smas[i-1]
        
        # HYSTERESIS: Only switch if trend breaks
        if prev_price < prev_sma:
            state = 0.0 # Defensive (Cash)
        else:
            state = 1.0 # Aggressive (Market)
            
        # Calculate Return
        daily_ret = (prices[i] - prices[i-1]) / prices[i-1]
        
        # Apply Logic
        if state == 1.0:
            new_eq = equity[-1] * (1 + daily_ret)
        else:
            new_eq = equity[-1] # Cash holds value (0% change)
            
        equity.append(new_eq)
        states.append(state)
        
    df['Protected_Equity'] = equity
    df['System_State'] = states
    
    # 4. Calculate Drawdowns
    df['Protected_DD'] = (df['Protected_Equity'] - df['Protected_Equity'].cummax()) / df['Protected_Equity'].cummax()
    
    # 5. GENERATE VIZ (Plotly)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=("Equity: Nifty 50 (Grey) vs Watchdog (Green)", "System State (Red = LIQUIDATED)"),
                        row_heights=[0.7, 0.3])
    
    # Equity Curve
    fig.add_trace(go.Scatter(x=df.index, y=df['Close']/df['Close'].iloc[0]*100, 
                             name="Nifty 50 (Buy & Hold)", line=dict(color='grey', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Protected_Equity'], 
                             name="Watchdog Protected", line=dict(color='#00ffcc', width=2)), row=1, col=1)
    
    # Regime State
    fig.add_trace(go.Scatter(x=df.index, y=df['System_State'], 
                             name="Regime (1=On, 0=Off)", fill='tozeroy', line=dict(color='#ff5555')), row=2, col=1)
    
    fig.update_layout(template="plotly_dark", title="ARCHITECT WATCHDOG AUDIT")
    fig.write_html("watchdog_audit.html")
    print("Visualization Generated: watchdog_audit.html")
    
    # 6. METRICS
    nifty_dd = df['Drawdown'].min()
    prot_dd = df['Protected_DD'].min()
    
    print("\n--- WATCHDOG METRICS ---")
    print(f"Nifty Max Drawdown:    {nifty_dd:.2%}")
    print(f"Protected Max Drawdown:{prot_dd:.2%}")
    if prot_dd > nifty_dd + 0.10: # If we saved 10%
        print("VERDICT: SYSTEM EFFECTIVE.")
    else:
        print("VERDICT: LAG DETECTED. (Optimization Required)")

if __name__ == "__main__":
    run_diagnosis()
