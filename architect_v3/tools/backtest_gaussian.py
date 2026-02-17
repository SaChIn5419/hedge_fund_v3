import pandas as pd
import numpy as np
import os
import sys
import glob

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import SignalProcessor
try:
    from strategies.gaussian_channel.alpha_engine import SignalProcessor
except ImportError:
    # If specific import fails, try direct file load or path hack
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../strategies/gaussian_channel')))
    from alpha_engine import SignalProcessor

def run_backtest(start_date="2020-01-01", end_date="2099-12-31", capital=1000000):
    print(f"--- GAUSSIAN CHANNEL BACKTEST ({start_date} - Present) ---")
    
    data_dir = "data/nse" # Detected from list_dir (assumed)
    if not os.path.exists(data_dir):
        # Fallback
        data_dir = "data/raw"
    
    files = glob.glob(f"{data_dir}/*.parquet")
    print(f"Found {len(files)} files in {data_dir}")
    
    all_trades = []
    
    for file in files:
        ticker = os.path.basename(file).replace(".parquet", "")
        # Filter exclusions if any (e.g. Indices)
        if ticker.startswith("^") or "NIFTY" in ticker:
            continue
            
        try:
            df = pd.read_parquet(file)
            # Standardize columns
            df.columns = [c.lower() for c in df.columns]
            
            # Filter Date
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
            
            if len(df) < 50: continue
            
            # Generate Signals
            df = SignalProcessor.generate_signals(df)
            
            # simulate logic: 
            # Buy if Bull_Signal == 1 (Signal from yesterday, buy today open)
            # Sell if Bear_Signal == -1 (Signal from yesterday, sell today open)
            
            # Allow skipping logic loop by vectorizing
            # Signal is on 'close' of T. Buy is on 'open' of T+1.
            
            df['signal'] = 0
            df.loc[df['Bull_Signal'] == 1, 'signal'] = 1
            df.loc[df['Bear_Signal'] == 1, 'signal'] = -1 # Assuming Bear_Signal is 1 for Sell?
            # alpha_engine says: Bear_Signal = np.where(..., -1, 0)
            df.loc[df['Bear_Signal'] == -1, 'signal'] = -1
            
            # Shift signal to represent "Action at T"
            # Signal generated at T-1 Close. Action at T Open.
            df['action'] = df['signal'].shift(1).fillna(0)
            
            # Iterate for positions
            in_position = False
            entry_price = 0
            entry_date = None
            
            # Simple Loop for accuracy with state
            dates = df['date'].values
            opens = df['open'].values
            closes = df['close'].values
            actions = df['action'].values
            
            for i in range(len(df)):
                date = dates[i]
                price = opens[i] # Trade at Open
                action = actions[i]
                
                if not in_position and action == 1:
                    in_position = True
                    entry_price = price
                    entry_date = date
                elif in_position and action == -1:
                    in_position = False
                    exit_price = price
                    pnl_pct = (exit_price - entry_price) / entry_price
                    # Log trade
                    all_trades.append({
                        'ticker': ticker,
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': pnl_pct
                    })
                    
        except Exception as e:
            # print(f"Error {ticker}: {e}")
            pass
            
    print(f"Total Trades Generated: {len(all_trades)}")
    
    if len(all_trades) == 0:
        print("No trades generated.")
        return

    trades_df = pd.DataFrame(all_trades)
    
    # Generate Equity Curve
    # Sort by exit_date (for realized PnL)
    trades_df = trades_df.sort_values('exit_date')
    
    # We need a daily series for the tearsheet.
    # This requires constructing a daily portfolio value.
    # Approximate: Start with Capital, apply trade returns sequentially?
    # Or parallel? 
    # Gaussian Channel Main Executor uses "Capital Per Position".
    # Let's assume Fixed Fractional or Fixed Capital. 
    # For tearsheet simplicity: Assume 100% committed to strategy, 
    # OR combine daily PnL of all trades.
    
    # Let's assume we allocated capital/N per trade.
    # But to get a "Strategy Equity Curve", we can just sum PnL.
    
    # Let's create a proxy daily equity curve
    # 10 Lakh starting. 
    # PnL = return * 1 Lakh (Mock size)
    
    trades_df['net_pnl'] = trades_df['return'] * (capital / 5) # 5 Positions max assumption
    
    # Aggregate to daily
    daily_pnl = trades_df.groupby('exit_date')['net_pnl'].sum().reset_index()
    daily_pnl['date'] = daily_pnl['exit_date']
    daily_pnl = daily_pnl.sort_values('date')
    
    daily_pnl['equity'] = daily_pnl['net_pnl'].cumsum() + capital
    
    # Save for tearsheet
    daily_pnl.to_csv("data/gaussian_backtest_equity.csv", index=False)
    trades_df.to_csv("data/gaussian_trades.csv", index=False)
    
    # Run Analytics
    import polars as pl
    from engine.analytics import PolarsTearSheet
    
    date_map = daily_pnl[['date', 'equity']].copy()
    date_map['date'] = pd.to_datetime(date_map['date'])
    
    # Fill missing dates for smooth curve
    idx = pd.date_range(start_date, end_date)
    # Reindex logic... simplified:
    df_pl = pl.from_pandas(date_map)
    
    trades_pl = pl.from_pandas(trades_df).with_columns([
        pl.col('net_pnl').alias('pnl'),
        pl.col('return').alias('return_pct'),
    ])
    
    analytics = PolarsTearSheet()
    analytics.generate(df_pl, trades_pl)
    
    # Access and rename
    default_out = "architect_tearsheet_v10.html"
    final_out = "gaussian_tearsheet.html"
    if os.path.exists(default_out):
        os.rename(default_out, final_out)
        webbrowser.open('file://' + os.path.realpath(final_out))

if __name__ == "__main__":
    run_backtest()
