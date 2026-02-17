import pandas as pd
import numpy as np
import os

CAPITAL = 1000000

def get_sharpe(equity_curve):
    # Daily returns
    rets = equity_curve.pct_change().dropna()
    if rets.std() == 0: return 0.0
    return (rets.mean() / rets.std()) * np.sqrt(252)

def get_max_dd(equity_curve):
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return dd.min()

def get_cagr(equity_curve):
    if len(equity_curve) < 2: return 0.0
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if days == 0: return 0.0
    years = days / 365.25
    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/years) - 1

def process_architect(filepath, name):
    if not os.path.exists(filepath):
        print(f"Skipping {name}: File not found at {filepath}")
        return None
        
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Architect log has 'net_pnl'
    # Aggregate daily
    daily_pnl = df.groupby('date')['net_pnl'].sum()
    
    # Reconstruct Equity
    equity = daily_pnl.cumsum() + CAPITAL
    # Shift to start at Capital ? No, cumsum adds to capital.
    # Initial state: Day 0 = Capital.
    # But groupby date is trade dates.
    # Let's reindex to full date range? 
    # For now, trade dates are sufficient for relative comparison.
    
    # Add initial capital point
    start_date = daily_pnl.index[0] - pd.Timedelta(days=1)
    equity[start_date] = CAPITAL
    equity = equity.sort_index()
    
    return {
        'name': name,
        'equity': equity.iloc[-1],
        'return': (equity.iloc[-1] / CAPITAL) - 1,
        'cagr': get_cagr(equity),
        'sharpe': get_sharpe(equity),
        'max_dd': get_max_dd(equity),
        'trades': len(df)
    }

def process_chimera(filepath, name):
    if not os.path.exists(filepath):
        print(f"Skipping {name}: File not found at {filepath}")
        return None
        
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Chimera has 'weight' and 'fwd_return'
    # PnL = fwd_return * weight * CAPITAL (Fixed capital alloc model)
    # Or is it compounded? Chimera script seems to use fixed alloc for log: 
    # 'net_pnl': fwd_return * 100000 * final_weight
    # Wait, the script I saw used 100,000 multiplier. Let's stick to 'fwd_return' * 'weight' * CAPITAL
    
    # Re-calculate PnL to be consistent with 1M capital
    # But wait, Chimera might interpret weight as % of CURRENT equity.
    # Let's assume % of Initial Capital for simple comparison or assume Compounding.
    # Architect uses Compounding in `analytics.py` (cum_prod).
    # But the trade log export is usually raw PnL? 
    # Architect `trade_log` has `net_pnl` calculated as `net_return * initial_capital`.
    # So Architect log is FIXED fractional basis (always relative to initial).
    # BUT `analytics.py` calculates CAGR based on Portfolio Returns compounded.
    
    # To be fair: Calculate Daily Return % and compound it.
    
    # Architect T+0/T+1:
    # `net_return` column is per trade return on allocated capital.
    # `weight` is allocation.
    # Portfolio Return = Sum(net_return). (Since weight is already factored into net_return? No)
    # Architect code: 
    # (pl.col("weight") * pl.col("fwd_return")).alias("gross_return")
    # net_return = gross - cost.
    # So `net_return` IS the portfolio contribution (weighted return).
    # So Sum(net_return) per day is the Portfolio Daily Return.
    
    # Chimera:
    # `fwd_return` is asset return. `weight` is allocation.
    # Portfolio Return = Sum(weight * fwd_return) per day.
    
    df['trade_ret'] = df['weight'] * df['fwd_return']
    daily_ret = df.groupby('date')['trade_ret'].sum()
    
    # Compound
    equity = (1 + daily_ret).cumprod() * CAPITAL
    
    # Add start
    start_date = daily_ret.index[0] - pd.Timedelta(days=1)
    equity[start_date] = CAPITAL
    equity = equity.sort_index()
    
    return {
        'name': name,
        'equity': equity.iloc[-1],
        'return': (equity.iloc[-1] / CAPITAL) - 1,
        'cagr': get_cagr(equity),
        'sharpe': get_sharpe(equity),
        'max_dd': get_max_dd(equity),
        'trades': len(df)
    }

def run():
    metrics = []
    
    # Architect
    m_t0 = process_architect("trade_log_t0.csv", "Architect V10 (T+0)")
    if m_t0: metrics.append(m_t0)
    
    m_t1 = process_architect("trade_log_t1.csv", "Architect V10 (T+1)")
    if m_t1: metrics.append(m_t1)
    
    # Chimera
    m_chi = process_chimera("data/chimera_blackbox_final.csv", "Chimera Final")
    if m_chi: metrics.append(m_chi)
    
    print("\n| Strategy | Total Return | CAGR | Sharpe | Max DD | Final Equity | Trades |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for m in metrics:
        print(f"| {m['name']} | {m['return']*100:,.2f}% | {m['cagr']*100:.2f}% | {m['sharpe']:.2f} | {m['max_dd']*100:.2f}% | {m['equity']:,.0f} | {m['trades']} |")

if __name__ == "__main__":
    run()
