import pandas as pd
import numpy as np

def calculate_metrics(df, capital=1000000):
    # Chimera log has 'net_pnl' per trade. 
    # We need to reconstruct equity curve to get MaxDD and Sharpe.
    # The log format from strategy_chimera_final.py:
    # date, ticker, close, weight, kinetic_energy, efficiency, ..., net_pnl
    
    # Aggregate PnL by Date
    df['date'] = pd.to_datetime(df['date'])
    daily_pnl = df.groupby('date')['net_pnl'].sum()
    
    # Reconstruct Equity
    # Assuming initial capital 1M
    equity = [capital]
    dates = [daily_pnl.index[0]] # approximation
    
    # We need a continuous date range for Sharpe? 
    # Or just trade dates? The strategy rebalances weekly.
    # Let's use the trade dates.
    
    current_eq = capital
    equity_curve = []
    
    for date, pnl in daily_pnl.items():
        current_eq += pnl
        equity_curve.append({'date': date, 'equity': current_eq})
        
    eq_df = pd.DataFrame(equity_curve).set_index('date')
    
    # Metrics
    if eq_df.empty:
        return "No Trades"
        
    final_eq = eq_df['equity'].iloc[-1]
    total_ret = (final_eq / capital) - 1
    
    # Duration in years
    duration = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
    cagr = (final_eq / capital) ** (1/duration) - 1 if duration > 0 else 0
    
    # Sharpe (approximate using trade intervals if not daily)
    # Be careful: if weekly, we multiply by sqrt(52).
    # Let's infer frequency.
    diffs = eq_df.index.to_series().diff().mean().days
    if diffs < 2:
        annual_factor = 252 # Daily
    elif diffs < 10:
        annual_factor = 52 # Weekly
    else:
        annual_factor = 12 # Monthly
        
    rets = eq_df['equity'].pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(annual_factor) if rets.std() > 0 else 0
    
    # Max Drawdown
    peak = eq_df['equity'].cummax()
    dd = (eq_df['equity'] - peak) / peak
    max_dd = dd.min()
    
    print(f"--- CHIMERA METRICS ({len(df)} Trades) ---")
    print(f"Final Equity: {final_eq:,.2f}")
    print(f"Total Return: {total_ret*100:.2f}%")
    print(f"CAGR: {cagr*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    print(f"Win Rate: {(len(df[df['net_pnl']>0])/len(df))*100:.2f}%")

if __name__ == "__main__":
    try:
        df = pd.read_csv("data/chimera_blackbox_final.csv")
        calculate_metrics(df)
    except Exception as e:
        print(f"Error: {e}")
