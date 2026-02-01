import os
import sys
import glob
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to allow importing modules from legacy_modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from legacy_modules.core.data_loader import DataEngine
except ImportError:
    print("Error: Could not import DataEngine. Ensure 'legacy_modules' is in the python path.")
    sys.exit(1)

def get_ticker_from_path(path):
    basename = os.path.basename(path)
    return basename.replace('.parquet', '')

def update_ticker(engine, ticker):
    try:
        # print(f"Updating {ticker}...")
        success = engine.update_dataset(ticker)
        return success
    except Exception as e:
        print(f"Failed to update {ticker}: {e}")
        return False

def main():
    print("ARCHITECT V3: Data Update Utility")
    print("-----------------------------------")
    
    # Path to data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'nse')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    # Initialize Engine
    # Note: DataEngine defaults to data/raw, but we want architect_v3/data/nse
    # Wait, existing files are in data/nse. 
    # DataEngine handles storage_dir in init.
    engine = DataEngine(storage_dir=data_dir)

    # Get list of existing tickers
    files = glob.glob(os.path.join(data_dir, "*.parquet"))
    tickers = [get_ticker_from_path(f) for f in files]
    
    # limit for verification (optional, remove limit for full run)
    # tickers = tickers[:5] # Un-comment to test with small set
    
    print(f"Found {len(tickers)} tickers in {data_dir}")
    print("Starting parallel update (This may take a while)...")

    # Use ThreadPoolExecutor for IO-bound task (network requests)
    # 4-8 threads is usually good for yfinance to avoid rate limits
    max_workers = 8 
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm for progress bar
        list(tqdm(executor.map(lambda t: update_ticker(engine, t), tickers), total=len(tickers), unit="ticker"))

    print("\nData update complete.")

if __name__ == "__main__":
    main()
