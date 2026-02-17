import requests
import json
import os

# Tickers from Chimera Engine
TICKERS = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
    'TATAMOTORS', 'SBIN', 'BAJFINANCE', 'ASIANPAINT', 'TITAN',
    'SUNPHARMA', 'JSWSTEEL', 'NTPC', 'POWERGRID', 'ITC',
    'ADANIENT', 'KPITTECH', 'ZOMATO', 'HAL', 'TRENT',
    'BSE', 'CDSL', 'MCX', 'ANGELONE', 'GOLDBEES'
]

# Indices
INDICES = {
    'NIFTY 50': 'Nifty 50', 
    'NIFTY BANK': 'Nifty Bank'
}

def fetch_tokens():
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    print(f"Downloading Master Contract from {url}...")
    
    try:
        r = requests.get(url)
        data = r.json()
        
        token_map = {}
        
        print(f"Processing {len(data)} records...")
        
        for item in data:
            # We want NSE Equity
            if item['exch_seg'] == 'NSE':
                symbol = item['symbol']
                token = item['token']
                name = item['name']
                
                # Check against our list (Exact match or close?)
                # Angel One symbols often have -EQ suffix for some, but usually just the name in 'name' or 'symbol'
                # 'symbol' in JSON is usually like 'RELIANCE-EQ'
                
                clean_symbol = symbol.replace('-EQ', '').strip()
                
                if clean_symbol in TICKERS:
                    token_map[clean_symbol] = {
                        'token': token,
                        'symbol': symbol,
                        'exch_seg': 'NSE'
                    }
                    
                # Indices (often in 'NSE' or 'NFO' but underlying index token is needed for websocket sometimes)
                # Actually for Indices, exch_seg is 'NSE' and symbol is like 'Nifty 50'
                if name in INDICES.values() and item['instrumenttype'] == 'AMXIDX':
                     token_map[name] = {
                        'token': token,
                        'symbol': symbol,
                        'exch_seg': 'NSE'
                    }

        print(f"Mapped {len(token_map)} tokens.")
        
        # Save
        os.makedirs("data", exist_ok=True)
        with open("data/angel_token_map.json", "w") as f:
            json.dump(token_map, f, indent=4)
            
        print("Saved to data/angel_token_map.json")
        return token_map
        
    except Exception as e:
        print(f"Error fetching tokens: {e}")
        return {}

if __name__ == "__main__":
    fetch_tokens()
