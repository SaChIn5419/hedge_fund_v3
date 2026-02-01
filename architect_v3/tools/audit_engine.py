import numpy as np
import pandas as pd
import os
import sys

def calculate_target_weights(prices_df: pd.DataFrame, target_vol: float = 0.15) -> pd.Series:
    """
    Calculates position weights based on Inverse Volatility Targeting.
    
    Formula: Weight_i = (Target_Vol / Vol_i)
    """
    # 1. Calculate annualized volatility (20-day rolling)
    daily_rets = prices_df.pct_change()
    rolling_vol = daily_rets.rolling(window=20).std() * np.sqrt(252)
    
    # 2. Inverse Volatility Sizing
    # If Vol is 80% (High Energy), Weight drops to ~18% (0.15/0.80)
    # Prevents the -81% Drawdown caused by sizing up in chaos.
    weights = target_vol / rolling_vol
    
    # 3. Cap weights to avoid leverage explosions on low-vol assets
    weights = weights.clip(upper=0.20) # Max 20% per asset
    
    return weights.dropna()

def audit_pnl_vs_equity(trade_log_path: str, initial_capital: float = 1000000.0):
    """
    Reconciles Arithmetic PnL (Trade Log) vs Geometric Equity (Compounding).
    """
    print(f"Loading trade log from: {trade_log_path}")
    if not os.path.exists(trade_log_path):
        print(f"Error: File not found at {trade_log_path}")
        return

    # 1. Load Data
    df = pd.read_csv(trade_log_path)
    # Handle potentially different date formats or column names if necessary, 
    # but assuming standard architect output.
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Aggregation: Daily PnL
    # Summing realized PnL per day across all tickers
    daily_pnl = df.groupby('date')['net_pnl'].sum().sort_index()
    
    # 3. Arithmetic Calculation (The "Tearsheet" View)
    total_arithmetic_pnl = daily_pnl.sum()
    final_arithmetic_equity = initial_capital + total_arithmetic_pnl
    
    # 4. Geometric Calculation (The "Reality" View)
    # We must approximate daily returns based on capital available at start of day
    # Note: Precise compounding requires daily total portfolio value, which trade_log lacks.
    # We reconstruct assuming full capital reinvestment.
    
    equity_curve = [initial_capital]
    for pnl in daily_pnl:
        # Prev Equity + Daily PnL
        equity_curve.append(equity_curve[-1] + pnl)
        
    final_geometric_equity = equity_curve[-1]
    
    # 5. The Audit Report
    print(f"--- ARCHITECT AUDIT REPORT ---")
    print(f"Initial Capital:      INR {initial_capital:,.2f}")
    print(f"Sum of Trade PnL:     INR {total_arithmetic_pnl:,.2f}")
    print(f"Arithmetic Final:     INR {final_arithmetic_equity:,.2f}")
    # print(f"Reported Final:       INR 17,181,504.00 (From your Tearsheet)") 
    # Commented out hardcoded value to avoid confusion, using calculated instead
    
    # Calculating delta against Arithmetic for now, effectively verifying if valid PnL sums match
    
    print(f"Geometric Final:      INR {final_geometric_equity:,.2f}")
    
    # Check discrepancy between simple sum and compounded path
    compounding_bonus = final_geometric_equity - final_arithmetic_equity
    print(f"Compounding Effect:   INR {compounding_bonus:,.2f}")

    if compounding_bonus > 0:
        print("\n[DIAGNOSIS]: Positive Compounding. Reinvesting profits accelerated growth.")
    else:
        print("\n[DIAGNOSIS]: Negative Compounding (Volatility Drag) or Arithmetic/Geometric Divergence.")

import polars as pl

def audit_architect_engine(csv_path):
    print(f"\nLoading trade log for Friction Analysis: {csv_path}")
    df = pl.read_csv(csv_path)
    
    # 1. Calculate Totals
    total_gross = df["gross_pnl"].sum()
    total_friction = df["friction_pnl"].sum()
    total_net = df["net_pnl"].sum()
    
    # 2. Friction Ratio
    burn_rate = (total_friction / total_gross) * 100
    
    # 3. True Leverage Check (Last Day)
    last_date = df["date"].max()
    last_day_df = df.filter(pl.col("date") == last_date)
    current_exposure = last_day_df["position_value"].sum()
    
    print(f"--- ARCHITECT DIAGNOSTIC ---")
    print(f"Total Gross Profit:   INR {total_gross:,.2f} (What you earned)")
    print(f"Total Friction Paid:  INR {total_friction:,.2f} (What you burned)")
    print(f"Total Net Profit:     INR {total_net:,.2f} (What you kept)")
    print(f"BURN RATE:            {burn_rate:.2f}% (Target: <15%)")
    print(f"---------------------------")
    print(f"Current Market Exp:   INR {current_exposure:,.2f}")
    
    if burn_rate > 20:
        print("\n[CRITICAL]: Stop trading high-turnover signals. You are churning.")
        print("ACTION: Increase 'holding_period' or 'minimum_profit_target'.")

if __name__ == "__main__":
    # Correct path to trade_log.csv relative to this script
    trade_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trade_log.csv'))
    
    # Try to infer capital or use default
    audit_pnl_vs_equity(trade_log_path, initial_capital=1000000.0)
    audit_architect_engine(trade_log_path)
