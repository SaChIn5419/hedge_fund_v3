import json
import os
import glob

def build_map():
    print("--- BUILDING MASTER TOKEN MAP ---")
    
    # 1. Load Angel Scrip Master
    master_path = "data/OpenAPIScripMaster.json"
    if not os.path.exists(master_path):
        print(f"ERROR: {master_path} not found.")
        return

    print("Loading Scrip Master...")
    with open(master_path, 'r') as f:
        master_data = json.load(f)
        
    # Create lookup: Symbol -> Token info
    # Filter for NSE Equity only to speed up
    angel_map = {}
    print(f"Processing {len(master_data)} master records...")
    for item in master_data:
        if item.get('exch_seg') == 'NSE' and '-EQ' in item.get('symbol', ''):
            # Angel Symbol format: "RELIANCE-EQ"
            # Our Lake format: "RELIANCE" (from RELIANCE.NS.parquet)
            clean_sym = item['symbol'].replace('-EQ', '')
            angel_map[clean_sym] = {
                'token': item['token'],
                'symbol': item['symbol'],
                'name': item['name'],
                'exch_seg': item['exch_seg']
            }
            
    print(f"Indexed {len(angel_map)} NSE Equity tokens.")

    # 2. Scan Data Lake
    lake_path = "data/nse"
    lake_files = glob.glob(os.path.join(lake_path, "*.parquet"))
    print(f"Found {len(lake_files)} files in Data Lake.")
    
    final_map = {}
    missing_count = 0
    
    for filepath in lake_files:
        filename = os.path.basename(filepath)
        # "RELIANCE.NS.parquet" -> "RELIANCE"
        ticker = filename.replace(".NS.parquet", "").replace(".parquet", "")
        
        # Handle special cases (e.g. ^NSEI)
        if ticker.startswith("^"):
            # Indices might need different mapping or manual handling
            # Angel Indices are usually in 'NSE' segment but might have different symbols
            # For now, skip indices or add manual mapping if critical
            continue
            
        if ticker in angel_map:
            final_map[ticker] = angel_map[ticker]
        else:
            # Try fuzzy match or known variations?
            # e.g. M&M vs M_M
            special_ticker = ticker.replace('&', '')
            if special_ticker in angel_map:
                 final_map[ticker] = angel_map[special_ticker]
            else:
                # print(f"Missing Token for: {ticker}")
                missing_count += 1

    # 3. Save Map
    out_path = "data/angel_token_map.json"
    with open(out_path, 'w') as f:
        json.dump(final_map, f, indent=2)
        
    print(f"success: Mapped {len(final_map)} tickers.")
    print(f"warning: Could not map {missing_count} tickers (likely Indices or De-listed).")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    build_map()
