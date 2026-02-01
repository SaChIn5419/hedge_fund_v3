# Architect_Crypto/visualize_strategy.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration matching the Alpha Model
LENGTH = 30
MULT = 1.5
CRYPTO_DATA_DIR = "crypto_data"
ASSETS = ['BTC-USD', 'ETH-USD', 'SOL-USD']

def calculate_gaussian_channel(df, length=30, mult=1.5):
    """
    Exact replication of CryptoGaussianModel logic.
    """
    df = df.copy()
    
    # GAUSSIAN MATH
    alpha = 2 / (length + 1)
    beta = 1 - alpha
    src = df['Close']
    filt = np.zeros(len(src))
    
    # 4th Order Gaussian Filter
    # Note: The loop in the original code started at index 4. 
    # We replicate that behavior. 
    # To be perfectly identical, we must iterate.
    # Vectorization is hard for recursive filters, so we stick to the loop.
    src_values = src.values
    for i in range(4, len(src)):
        filt[i] = (alpha**4)*src_values[i] + \
                  4*beta*filt[i-1] - \
                  6*(beta**2)*filt[i-2] + \
                  4*(beta**3)*filt[i-3] - \
                  (beta**4)*filt[i-4]
                  
    df['Gaussian_Filt'] = filt
    
    # TRUE RANGE BANDS
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
    df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['FTR'] = df['TR'].rolling(window=length).mean()
    
    df['Gaussian_Upper'] = df['Gaussian_Filt'] + (df['FTR'] * mult)
    df['Gaussian_Lower'] = df['Gaussian_Filt'] - (df['FTR'] * mult)
    
    # KINETIC ENERGY (Volume)
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    df['High_Energy'] = df['Volume'] > df['Vol_MA']
    
    return df

def plot_strategy(ticker, df):
    print(f"Plotting {ticker}...")
    
    # Filter for plot readability - last 2 years or full?
    # Let's do the last 2 years to see recent behavior explicitly
    plot_df = df.loc['2022-01-01':].copy()
    
    if plot_df.empty:
        print(f"Not enough data for {ticker} since 2022.")
        plot_df = df.copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # PRICE & BANDS
    ax1.plot(plot_df.index, plot_df['Close'], label='Price', color='black', alpha=0.7, linewidth=1)
    ax1.plot(plot_df.index, plot_df['Gaussian_Upper'], label='Upper Band', color='green', linestyle='--', alpha=0.6)
    ax1.plot(plot_df.index, plot_df['Gaussian_Lower'], label='Lower Band', color='red', linestyle='--', alpha=0.6)
    ax1.plot(plot_df.index, plot_df['Gaussian_Filt'], label='Gaussian Filter', color='blue', alpha=0.3)
    
    # Fill channel
    ax1.fill_between(plot_df.index, plot_df['Gaussian_Upper'], plot_df['Gaussian_Lower'], color='gray', alpha=0.1)

    # Highlight ENTRY points
    # Logic: Close > Upper Band AND High Energy
    entries = plot_df[(plot_df['Close'] > plot_df['Gaussian_Upper']) & (plot_df['High_Energy'])]
    ax1.scatter(entries.index, entries['Close'], color='lime', marker='^', s=100, label='Long Signal', zorder=5)
    
    # Highlight EXIT points
    # Logic: Close < Lower Band
    exits = plot_df[plot_df['Close'] < plot_df['Gaussian_Lower']]
    ax1.scatter(exits.index, exits['Close'], color='red', marker='v', s=100, label='Exit Signal', zorder=5)

    ax1.set_title(f"Architect Strategy Inspection: {ticker} (Gaussian Ch 30, 1.5x) + Volume Energy")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # VOLUME
    colors = np.where(plot_df['High_Energy'], 'lime', 'gray')
    ax2.bar(plot_df.index, plot_df['Volume'], color=colors, alpha=0.5, label='Volume')
    ax2.plot(plot_df.index, plot_df['Vol_MA'], color='orange', label='Vol MA (20)')
    
    ax2.set_title("Volume Energy (Green = High Energy)")
    ax2.set_ylabel("Volume")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{ticker}_strategy_debug.png"
    plt.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close()

def main():
    for ticker in ASSETS:
        file_path = os.path.join(CRYPTO_DATA_DIR, f"{ticker}.parquet")
        if not os.path.exists(file_path):
            print(f"Skipping {ticker}, file not found.")
            continue
            
        df = pd.read_parquet(file_path)
        # Ensure correct column names
        df.columns = ['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df.set_index('Date', inplace=True)
        
        # Calculate Indicators
        df_calc = calculate_gaussian_channel(df, LENGTH, MULT)
        
        # Plot
        plot_strategy(ticker, df_calc)

if __name__ == "__main__":
    main()
