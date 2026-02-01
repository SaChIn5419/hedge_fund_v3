# Architect_Crypto/crypto_vector_backtest.py
import pandas as pd
import numpy as np
import duckdb
import os
import quantstats as qs

# -----------------------------------------------------------------------------
# 1. THE PHYSICS LAYER (Ehlers Gaussian IIR Filter)
# -----------------------------------------------------------------------------
def calc_gaussian_filter(src, length=30):
    """
    Calculates the 4-Pole Gaussian Filter. 
    Unlike simple Moving Averages, this is an Infinite Impulse Response (IIR) filter.
    """
    alpha = 2 / (length + 1)
    beta = 1 - alpha
    src_arr = src.values
    filt = np.zeros(len(src_arr))
    
    # Fast Numpy Loop for Recursive Math
    c0 = alpha**4
    c1 = 4 * beta
    c2 = -6 * (beta**2)
    c3 = 4 * (beta**3)
    c4 = -(beta**4)

    for i in range(4, len(src_arr)):
        filt[i] = c0 * src_arr[i] + c1 * filt[i-1] + c2 * filt[i-2] + c3 * filt[i-3] + c4 * filt[i-4]
        
    return pd.Series(filt, index=src.index)

# -----------------------------------------------------------------------------
# 2. THE VECTORIZED BACKTESTER (Per-Asset)
# -----------------------------------------------------------------------------
def run_vectorized_backtest(df, ticker, length=30, mult=1.5, fee_pct=0.0015):
    """
    Simulates the strategy for a single coin using pure matrix math.
    fee_pct = 0.10% Binance Fee + 0.05% Slippage = 0.0015
    """
    # 1. GAUSSIAN BANDS
    df['Gaussian_Filt'] = calc_gaussian_filter(df['close'], length)
    df['High_Low'] = df['high'] - df['low']
    df['High_Close'] = np.abs(df['high'] - df['close'].shift())
    df['Low_Close'] = np.abs(df['low'] - df['close'].shift())
    df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['FTR'] = df['TR'].rolling(window=length).mean()
    
    df['Upper_Band'] = df['Gaussian_Filt'] + (df['FTR'] * mult)
    df['Lower_Band'] = df['Gaussian_Filt'] - (df['FTR'] * mult)

    # 2. KINETIC ENERGY (Volume > 70% of 20-Day Average)
    df['Vol_MA'] = df['volume'].rolling(window=20).mean()
    df['High_Energy'] = df['volume'] > (df['Vol_MA'] * 0.7)

    # 3. VECTORIZED SIGNALS
    # 1 = In the Market (Long), 0 = In Cash
    # We buy when Price > Upper Band + Energy. We sell when Price < Lower Band.
    df['Signal'] = np.nan
    df.loc[(df['close'] > df['Upper_Band']) & (df['High_Energy']), 'Signal'] = 1
    df.loc[df['close'] < df['Lower_Band'], 'Signal'] = 0
    df['Signal'] = df['Signal'].ffill().fillna(0) # Hold the state until a new signal

    # 4. VECTORIZED RETURNS & FEES
    df['Asset_Returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # We shift the signal by 1 day to simulate buying on the NEXT open
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Asset_Returns']
    
    # Calculate Trades (Every time signal changes from 0 to 1 or 1 to 0)
    df['Trades'] = df['Signal'].diff().abs()
    
    # Subtract Fees (Binance 0.1% + 0.05% Slippage) per trade
    df['Strategy_Returns'] -= df['Trades'] * fee_pct

    # -------------------------------------------------------------------------
    # 5. TRADE LOGGING (Reconstruct Individual Trades)
    # -------------------------------------------------------------------------
    trades = []
    in_trade = False
    entry_date = None
    entry_price = 0.0
    
    # Iterate to find trade points (Vectorized is fast, but logging needs loop or smart grouping)
    # We can use the 'Trades' column which identifies changes
    # 0->1 : Buy (Entry) at Close of day
    # 1->0 : Sell (Exit) at Close of day
    
    # Identify Signal changes
    df['Sig_Diff'] = df['Signal'].diff()
    
    entries = df[df['Sig_Diff'] == 1.0]
    exits = df[df['Sig_Diff'] == -1.0]
    
    # We match entries and exits chronologically
    # Note: Logic assumes we always exit before next enter (which Signal 1/0 enforces)
    
    # To be robust, let's iterate the changes
    trade_events = df[df['Sig_Diff'] != 0].dropna()
    
    for date, row in trade_events.iterrows():
        if row['Sig_Diff'] == 1.0: # Buy Signal
            entry_date = date
            entry_price = row['close']
            in_trade = True
        elif row['Sig_Diff'] == -1.0 and in_trade: # Sell Signal
            exit_date = date
            exit_price = row['close']
            
            pnl = (exit_price - entry_price) / entry_price
            pnl -= (fee_pct * 2) # Fee on entry and exit math approx
            
            duration = (exit_date - entry_date).days
            
            trades.append({
                'Ticker': ticker,
                'Entry_Date': entry_date,
                'Exit_Date': exit_date,
                'Duration_Days': duration,
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'PnL_Pct': round(pnl * 100, 2),
                'PnL_Abs': round(exit_price - entry_price, 4)
            })
            in_trade = False
            
    # Handle open trade at end
    if in_trade:
        current_price = df['close'].iloc[-1]
        pnl = (current_price - entry_price) / entry_price
        trades.append({
            'Ticker': ticker,
            'Entry_Date': entry_date,
            'Exit_Date': pd.Timestamp.now(), # Still open
            'Duration_Days': (df.index[-1] - entry_date).days,
            'Entry_Price': entry_price,
            'Exit_Price': current_price,
            'PnL_Pct': round(pnl * 100, 2),
            'PnL_Abs': round(current_price - entry_price, 4),
            'Status': 'OPEN'
        })

    return df['Strategy_Returns'].fillna(0), trades

# -----------------------------------------------------------------------------
# 3. THE ORCHESTRATION ENGINE (Portfolio Aggregation)
# -----------------------------------------------------------------------------
def launch_crypto_grid():
    print("ARCHITECT: Booting Vectorized Crypto Grid...")
    
    # Load ALL Data instantly using DuckDB
    conn = duckdb.connect(':memory:')
    query = """
        SELECT ticker, date, open, high, low, close, volume 
        FROM 'crypto_data/*.parquet'
        ORDER BY date ASC
    """
    master_df = conn.execute(query).df()
    master_df['date'] = pd.to_datetime(master_df['date'])
    master_df.set_index('date', inplace=True)

    strategy_returns = []
    all_trades = []
    # Explicitly filter for the Trifecta
    TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD']

    # Run Backtest for each asset
    for ticker in TICKERS:
        print(f"Processing Matrix Math for: {ticker}")
        asset_df = master_df[master_df['ticker'] == ticker].copy()
        
        # Returns AND Trades
        strat_ret, asset_trades = run_vectorized_backtest(asset_df, ticker)
        
        strat_ret.name = ticker
        strategy_returns.append(strat_ret)
        all_trades.extend(asset_trades)

    # OUTPUT TRADE LOG
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.sort_values(by='Entry_Date', inplace=True)
        trades_df.to_csv("crypto_trade_log.csv", index=False)
        print(f"SUCCESS: Trade Log saved to crypto_trade_log.csv ({len(trades_df)} trades).")
    else:
        print("WARNING: No trades generated during the period.")

    # Combine into a single Equal-Weight Portfolio
    portfolio_df = pd.concat(strategy_returns, axis=1).fillna(0)
    portfolio_returns = portfolio_df.mean(axis=1) # Equal Weight

    # -----------------------------------------------------------------------------
    # 4. PERFORMANCE REPORTING (QuantStats)
    # -----------------------------------------------------------------------------
    # Convert log returns to simple returns for QuantStats
    simple_returns = np.exp(portfolio_returns) - 1

    # Fetch Benchmark (BTC) for comparison
    btc_df = master_df[master_df['ticker'] == 'BTC-USD'].copy()
    btc_returns = np.exp(np.log(btc_df['close'] / btc_df['close'].shift(1))) - 1
    
    print("\nARCHITECT: Generating Tear Sheet...")
    qs.reports.html(simple_returns, benchmark=btc_returns, title="Architect Crypto Arsenal (Vectorized)", output='crypto_vector_report.html')
    print("SUCCESS: crypto_vector_report.html generated.")

if __name__ == "__main__":
    launch_crypto_grid()
