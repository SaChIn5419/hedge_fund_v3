import pandas as pd
import polars as pl
import numpy as np
import sys
import os
import webbrowser

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chimera_analytics import PolarsTearSheet

def run_tearsheet():
    csv_path = "data/chimera_blackbox_final.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading Trade Log: {csv_path}")
    df_pd = pd.read_csv(csv_path)
    
    # Preprocess for Analytics
    # We need an Equity Curve DataFrame: date, equity
    
    # 1. Calculate Portfolio Return (Geometric Compounding)
    # The log contains 'fwd_return' (approx 1 week return) and 'weight'
    # Portfolio Return for that Period = Sum(Weight * Asset_Return) for all assets in that period
    
    # Check required columns
    if 'fwd_return' not in df_pd.columns or 'weight' not in df_pd.columns:
         print("Critical: fwd_return or weight missing. Cannot calc Equity.")
         return

    # Group by Date to get Period Return
    # df_pd['weighted_ret'] = df_pd['fwd_return'] * df_pd['weight']
    # If using leverage (sum of weights > 1), this correctly captures the leveraged return.
    
    period_returns = df_pd.groupby('date').apply(
        lambda x: (x['fwd_return'] * x['weight']).sum()
    ).reset_index(name='period_ret')
    
    period_returns = period_returns.sort_values('date')
    
    # 2. Reconstruct Equity (Compounding)
    capital = 1000000.0
    period_returns['equity'] = capital * (1 + period_returns['period_ret']).cumprod()
    
    # Calculate 'net_pnl' for visual reference in table (using actual capital at that time)
    # Re-merge equity back to trade log to calc PnL per trade based on DYNAMIC capital
    # Trade PnL = Return * Weight * Previous_Equity
    
    # We need previous equity. Shift equity by 1.
    period_returns['prev_equity'] = period_returns['equity'].shift(1).fillna(capital)
    
    # Merge 'prev_equity' back to main df
    df_pd = df_pd.merge(period_returns[['date', 'prev_equity']], on='date', how='left')
    df_pd['net_pnl'] = df_pd['fwd_return'] * df_pd['weight'] * df_pd['prev_equity']
    
    # Prepare Daily PnL for Analytics (It expects 'net_pnl' and 'equity' columns in the input DF?)
    # Analytics.generate takes 'df' which is the Equity Curve.
    # It expects: date, equity. It calculates 'ret' from equity.
    
    daily_pnl = period_returns[['date', 'equity', 'period_ret']]
    
    # 3. Create Polars DataFrame for Analytics
    # Ensure date is datetime
    daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
    
    # Add initial row
    start_date = daily_pnl['date'].iloc[0] - pd.Timedelta(days=1)
    initial_row = pd.DataFrame({'date': [start_date], 'equity': [capital], 'period_ret': [0.0]})
    daily_pnl = pd.concat([initial_row, daily_pnl]).sort_values('date').reset_index(drop=True)
    
    # Convert to Polars
    df_pl = pl.from_pandas(daily_pnl)
    
    # 4. Generate Tearsheet
    # We also need 'trades_df' for the dashboard list
    
    # Let's rename columns to match roughly
    trades_pl = pl.from_pandas(df_pd).with_columns([
        pl.col("date").alias("entry_date"), # Approx
        pl.col("fwd_return").alias("return_pct"),
        pl.col("net_pnl").alias("pnl")
    ])
    
    print("Generating Dashboard...")
    analytics = PolarsTearSheet(risk_free_rate=0.07, benchmark_ticker='^NSEI')
    
    if "holding_period" not in trades_pl.columns:
        trades_pl = trades_pl.with_columns(pl.lit(5).alias("holding_period"))
        
    # Run
    analytics.generate(df_pl, trades_pl)
    
    # The generate method saves to "architect_tearsheet_v10.html" by default
    default_output = "architect_tearsheet_v10.html"
    final_path = "chimera_v1.html"
    
    if os.path.exists(default_output):
        import shutil
        # Remove existing destination if any
        if os.path.exists(final_path):
            os.remove(final_path)
            
        shutil.move(default_output, final_path)
        print(f"Generated and Renamed to: {final_path}")
        webbrowser.open('file://' + os.path.realpath(final_path))
    else:
        print(f"Error: {default_output} was not generated.")

if __name__ == "__main__":
    run_tearsheet()
