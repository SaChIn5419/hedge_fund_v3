import yfinance as yf
import pandas as pd
import os

class DataEngine:
    def __init__(self, storage_dir="data/raw"):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def fetch_and_store(self, ticker, period="15y", interval="1d", allow_unsafe=False):
        """
        Fetches data from Yahoo Finance and validates it.
        If valid (or allow_unsafe=True), cleans and saves to CSV.
        """
        try:
            # Fetch data with strict check
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if data is None or data.empty:
                return False

            # Basic Validation
            is_valid, reason, cleaned_data = validate_data(data, ticker)
            
            if is_valid or allow_unsafe:
                if not is_valid:
                    print(f"   [WARNING] Data for {ticker} failed validation ({reason}) but forced saved.")
                    
                # Save to Parquet (Optimized)
                # Ensure Date is a column for DuckDB compatibility
                if cleaned_data.index.name != 'Date':
                    cleaned_data.index.name = 'Date'
                cleaned_data.reset_index(inplace=True)
                
                file_path = os.path.join(self.storage_dir, f"{ticker}.parquet")
                cleaned_data.to_parquet(file_path, index=False)
                return True
            else:
                # print(f"Invalid data for {ticker}: {reason}")
                return False
                
        except Exception as e:
            # print(f"Error fetching {ticker}: {e}")
            return False

    def update_dataset(self, ticker, allow_unsafe=False):
        """
        Smartly updates the dataset. 
        If file exists, fetches only missing data.
        If file missing, performs full download.
        """
        file_path = os.path.join(self.storage_dir, f"{ticker}.parquet")
        
        # 1. Check if file exists
        if not os.path.exists(file_path):
            print(f"   [NEW] Full Download for {ticker}...")
            return self.fetch_and_store(ticker, period="15y", interval="1d", allow_unsafe=allow_unsafe)
            
        try:
            # 2. Load existing data to find last date
            existing_df = pd.read_parquet(file_path)
            if 'Date' in existing_df.columns:
                existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                last_date = existing_df['Date'].max()
            else:
                # Fallback if Date is index
                if existing_df.index.name == 'Date':
                    existing_df.reset_index(inplace=True)
                    existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                    last_date = existing_df['Date'].max()
                else:
                    print(f"   [ERROR] Schema mismatch for {ticker}. Re-downloading.")
                    return self.fetch_and_store(ticker, period="15y", interval="1d", allow_unsafe=allow_unsafe)

            # 3. Calculate Start Date
            start_date = last_date + pd.Timedelta(days=1)
            today = pd.Timestamp.now().normalize()
            
            if start_date >= today:
                print(f"   [SKIP] {ticker} is up to date ({last_date.date()}).")
                return True
                
            print(f"   [UPDATE] Fetching {ticker} from {start_date.date()}...", end=" ")
            
            # 4. Fetch Delta
            new_data = yf.download(ticker, start=start_date, interval="1d", progress=False)
            
            if new_data is None or new_data.empty:
                print("No new data.")
                return True # Technically succeeded, just nothing new

            # 5. Validate Delta
            is_valid, reason, cleaned_new_data = validate_data(new_data, ticker)
            
            if not is_valid and not allow_unsafe:
                print(f"FAILED ({reason}).")
                return False
                
            # 6. Merge
            if cleaned_new_data.index.name != 'Date':
                cleaned_new_data.index.name = 'Date'
            cleaned_new_data.reset_index(inplace=True)
            
            # Ensure types match before concat
            cleaned_new_data['Date'] = pd.to_datetime(cleaned_new_data['Date'])
            
            # Concatenate
            full_df = pd.concat([existing_df, cleaned_new_data])
            
            # Deduplicate (Keep last)
            full_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            full_df.sort_values(by='Date', inplace=True)
            
            # Save
            full_df.to_parquet(file_path, index=False)
            print("DONE.")
            return True
            
        except Exception as e:
            print(f"   [ERROR] Update failed for {ticker}: {e}")
            return False

def validate_data(df, ticker=None):
    """
    Validates the integrity of financial data.
    Returns: (is_valid, reason, cleaned_df)
    """
    # Handle MultiIndex columns (common with recent yfinance)
    # If MultiIndex, drop the ticker level to simplify
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Drop the ticker level (level 1) if it exists and we only have one ticker
            df = df.droplevel(1, axis=1) 
        except Exception:
            pass # Keep as is if it fails

    if df.empty:
        return False, "Empty DataFrame", df

    # 1. Check for Zero Volume (Liquidity Trap)
    # Indices (^) often have 0 volume on Yahoo. Skip this check for them.
    if ticker and ticker.startswith("^"):
        pass
    elif 'Volume' in df.columns:
        vol = df['Volume']
        # Ensure vol is a Series
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
            
        if (vol == 0).sum() > len(df) * 0.1:
            return False, "Too many zero volume days", df
    
    # 2. Check for Flash Crash/Bad Ticks (Physics Filter)
    # Use Adj Close if available, else Close
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    if col in df.columns:
        prices = df[col]
        # Ensure prices is a Series
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
            
        daily_returns = prices.pct_change()
        if daily_returns.max() > 1.0 or daily_returns.min() < -0.9:
            return False, "Unrealistic price spike detected", df
        
    return True, "Clean", df
