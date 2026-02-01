import pandas as pd
import requests
import io

class UniverseScanner:
    def __init__(self):
        # Official NSE Live Market List URL
        self.nse_url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

    def fetch_nse_universe(self):
        """
        Scrapes the latest active equity list from NSE India.
        Returns a list of tickers formatted for Yahoo Finance (e.g., 'RELIANCE.NS')
        """
        print(f"[UNIVERSE] Connecting to NSE Archives...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(self.nse_url, headers=headers)
            if response.status_code == 200:
                csv_data = io.StringIO(response.content.decode('utf-8'))
                df = pd.read_csv(csv_data)
                
                # Filter for active equities (Series 'EQ')
                # We ignore BE (Book Entry) or SM (SME) to avoid illiquid traps initially
                df = df[df[' SERIES'] == 'EQ']
                
                # Format for Yahoo Finance: Symbol + ".NS"
                tickers = [f"{symbol}.NS" for symbol in df['SYMBOL'].values]
                
                print(f"[SUCCESS] Found {len(tickers)} active NSE equities.")
                return tickers
            else:
                print(f"[ERROR] NSE Server returned code: {response.status_code}")
                return []
        except Exception as e:
            print(f"[CRITICAL] Failed to fetch universe: {e}")
            return []

if __name__ == "__main__":
    scanner = UniverseScanner()
    tickers = scanner.fetch_nse_universe()
    print(f"Sample: {tickers[:5]}")
