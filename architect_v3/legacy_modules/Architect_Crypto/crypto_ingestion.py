# crypto_ingestion.py
import ccxt
import pandas as pd
import duckdb
import os
import time

# -----------------------------------------------------------------------------
# 1. THE TRIFECTA (BTC, ETH, SOL)
# -----------------------------------------------------------------------------
CRYPTO_TICKERS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT'
]

DATA_DIR = "crypto_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

exchange = ccxt.binance({
    'enableRateLimit': True,
})

def fetch_historical_data(symbol, start_year=2018):
    print(f"ARCHITECT: Connecting to Binance for {symbol}...")
    timeframe = '1d'
    # Fallback to 2018 for older coins, new coins will just start at their ICO date
    since = exchange.parse8601(f'{start_year}-01-01T00:00:00Z') 
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not len(ohlcv):
                break
            all_ohlcv += ohlcv
            since = ohlcv[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching {symbol} or Coin does not exist on Binance Spot: {e}")
            break

    if not all_ohlcv:
        return symbol.replace('/USDT', '-USD'), pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    clean_symbol = symbol.replace('/USDT', '-USD')
    df['ticker'] = clean_symbol
    df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
    
    return clean_symbol, df

def ingest_to_parquet():
    for coin in CRYPTO_TICKERS:
        ticker, df = fetch_historical_data(coin)
        
        if not df.empty:
            # Drop data older than the coin's actual listing to prevent zero-padding bugs
            df = df[df['close'] > 0] 
            parquet_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
            df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
            print(f"SUCCESS: {ticker} saved. Rows: {len(df)}")

if __name__ == "__main__":
    ingest_to_parquet()
