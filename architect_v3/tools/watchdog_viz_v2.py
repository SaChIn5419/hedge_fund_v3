import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
BENCHMARK_TICKER = '^NSEI'
TIMEFRAME = "15y" # Need 2008 and 2020 to test properly

def fetch_clean_data(ticker, period):
    print(f"WATCHDOG V2: Fetching {ticker} history ({period})...")
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Standardize column naming just in case
    if 'Close' not in df.columns:
        if len(df.columns) == 1:
            df.columns = ['Close']
    
    df.index = pd.to_datetime(df.index)
    return df

def run_fine_tuned_diagnosis():
    # 1. Ingest Data
    df = fetch_clean_data(BENCHMARK_TICKER, TIMEFRAME)
    if df.empty: return

    # 2. Compute Indicators
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Daily_Ret'] = df['Close'].pct_change()
    df['Drawdown'] = (df['Close'] - df['Close'].cummax()) / df['Close'].cummax()

    # 3. SIMULATION LOOP (Hybrid Trigger)
    # Start with 100 equity
    equity_slow = [100.0] # Old V8 Logic (SMA Only)
    equity_fast = [100.0] # New V9 Logic (SMA + Daily Drop)
    
    # Track states for visualization
    states_slow = [1.0]
    states_fast = [1.0]
    
    prices = df['Close'].values
    smas = df['SMA_200'].values
    rets = df['Daily_Ret'].values
    
    # Hysteresis trackers
    fast_mode_active = False # Are we in 'Panic Mode' from a drop?
    
    for i in range(1, len(df)):
        price = prices[i]
        sma = smas[i]
        ret = rets[i]
        
        # --- LOGIC 1: SLOW (OLD) ---
        # If Price < SMA, Go Cash.
        if price < sma:
            state_s = 0.0
        else:
            state_s = 1.0
            
        # Apply Return
        if states_slow[-1] == 1.0:
            eq_s = equity_slow[-1] * (1 + ret)
        else:
            eq_s = equity_slow[-1] # Cash
            
        # --- LOGIC 2: FAST (NEW) ---
        # Trigger A: Price < SMA (Trend is dead)
        # Trigger B: Daily Drop < -3% (Panic is here)
        
        # Check Trigger B (The Circuit Breaker)
        if ret < -0.03: 
            fast_mode_active = True
            
        # Check Reset Condition for Trigger B
        # We only reset if Price > SMA (Trend recovered) OR we see a strong bounce (+3%)
        # For safety, let's say we stay Defensive until Price > SMA 
        # (This is the "Hard Reset" rule)
        
        is_bear_trend = price < sma
        
        if fast_mode_active or is_bear_trend:
            state_f = 0.0
            
            # Can we turn it back on?
            # Only if Trend is Bullish AND we are not crashing today
            if not is_bear_trend and ret > -0.03:
                fast_mode_active = False
                state_f = 1.0
        else:
            state_f = 1.0
            
        # Apply Return (The Critical Fix)
        # If the Trigger fired TODAY (ret < -0.03), did we escape?
        # NO. We suffer TODAY'S loss, but we escape TOMORROW.
        # However, the SMA logic waits weeks. The Fast logic exits TOMORROW.
        
        if states_fast[-1] == 1.0:
             eq_f = equity_fast[-1] * (1 + ret)
        else:
             eq_f = equity_fast[-1] # Cash
             
        equity_slow.append(eq_s)
        equity_fast.append(eq_f)
        states_fast.append(state_f)

    df['Equity_Slow'] = equity_slow
    df['Equity_Fast'] = equity_fast
    df['State_Fast'] = states_fast
    
    # 4. Compute Metrics
    df['DD_Slow'] = (df['Equity_Slow'] - df['Equity_Slow'].cummax()) / df['Equity_Slow'].cummax()
    df['DD_Fast'] = (df['Equity_Fast'] - df['Equity_Fast'].cummax()) / df['Equity_Fast'].cummax()
    
    slow_dd_max = df['DD_Slow'].min()
    fast_dd_max = df['DD_Fast'].min()
    
    print("\n--- ARCHITECT FINE-TUNING RESULTS ---")
    print(f"Benchmark (Nifty) DD: {df['Drawdown'].min():.2%}")
    print(f"Old Logic (SMA Only) DD: {slow_dd_max:.2%} (The Lag)")
    print(f"New Logic (Hybrid) DD:   {fast_dd_max:.2%} (The Fix)")
    
    improvement = slow_dd_max - fast_dd_max
    print(f"Safety Improvement:     {improvement*100:.2f} percentage points")
    
    # 5. Visualize
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        subplot_titles=("Equity Curves: The Cost of Lag", "Circuit Breaker Activation"))
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Close']/df['Close'].iloc[0]*100, name="Nifty 50", line=dict(color='grey')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Equity_Slow'], name="Old Logic (Slow)", line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Equity_Fast'], name="New Logic (Fast)", line=dict(color='#00ffcc', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['State_Fast'], name="Fast Trigger", fill='tozeroy', line=dict(color='red')), row=2, col=1)
    
    fig.update_layout(template="plotly_dark", title="ARCHITECT V9: LAG ELIMINATION")
    fig.write_html("watchdog_tuning.html")
    print("Report Generated: watchdog_tuning.html")

if __name__ == "__main__":
    run_fine_tuned_diagnosis()
