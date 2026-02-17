import polars as pl
from SmartApi import SmartConnect
from datetime import datetime, timedelta
import os
import pyotp
import sys
import json
import logging
import time

# Add parent path to import config
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..')) # hedge_fund_v3
sys.path.insert(0, project_root)

# Credentials Loading
try:
    # 1. Try importing from Chimera_Pro_Live secrets
    secrets_path = os.path.join(project_root, 'Chimera_Pro_Live')
    if secrets_path not in sys.path:
        sys.path.append(secrets_path)
    
    from chimera_execution.secrets import API_KEY, CLIENT_ID, PASSWORD, TOTP_SECRET
    print("[OK] Loaded Credentials from Chimera_Pro_Live/chimera_execution/secrets.py")

except ImportError:
    try:
        # 2. Try root secrets.py if it exists
        from secrets import API_KEY, CLIENT_ID, PASSWORD, TOTP_SECRET
        print("[OK] Loaded Credentials from secrets.py")
    except ImportError:
        try:
            # 3. Fallback to standard config (might be empty/placeholders)
            from chimera_execution.config import API_KEY, CLIENT_ID, PASSWORD, TOTP_SECRET
            print("[WARN] Loaded Credentials from chimera_execution.config (Check if valid)")
        except ImportError as e:
            print(f"[ERROR] Failed to load credentials from secrets.py or config.py: {e}")
            sys.exit(1)

try:
    from chimera_protocol.chimera_engine import CONFIG
    print("[OK] Loaded CONFIG from chimera_protocol.chimera_engine")
except ImportError:
    print("[WARN]  CONFIG not found. Using default Tickers.")
    CONFIG = {'TICKERS': []}

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChimeraFoundry:
    def __init__(self, storage_path="data/nse"):
        self.storage_path = storage_path
        # Legacy structure had ohlcv subdir, but user's master lake is flat in data/nse
        # We will use storage_path directly for OHLCV
        self.ohlcv_path = storage_path 
        self.depth_path = os.path.join("data", "depth_lake") # Separate depth lake
        
        os.makedirs(self.ohlcv_path, exist_ok=True)
        os.makedirs(self.depth_path, exist_ok=True)
        
        self.api = SmartConnect(api_key=API_KEY)
        self.token_map = self.load_token_map()
        self.login()

    def load_token_map(self):
        path = os.path.join("data", "angel_token_map.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def get_token(self, symbol):
        clean = symbol.replace(".NS", "").strip()
        if clean in self.token_map:
            return self.token_map[clean]['token']
        return None

    def login(self):
        """Standard Authentication"""
        try:
            totp = pyotp.TOTP(TOTP_SECRET).now()
            data = self.api.generateSession(CLIENT_ID, PASSWORD, totp)
            if data['status']:
                self.jwt = data['data']['jwtToken']
                logger.info("[OK] Foundry Connected to Angel One")
            else:
                raise Exception(f"Foundry Auth Failed: {data['message']}")
        except Exception as e:
            logger.error(f"[ERROR] Auth Error: {e}")

    def fetch_recent_history(self, token, exchange="NSE", days=30, interval="ONE_DAY"):
        """
        Fetches the last N days to ensure we bridge any gaps.
        Angel One returns: [Timestamp, Open, High, Low, Close, Volume]
        """
        try:
            # Respect Rate Limits (2 requests per second is safe)
            time.sleep(0.5) 
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            params = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": interval,
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M")
            }
            
            response = self.api.getCandleData(params)
            if not response or not response.get('data'):
                logger.warning(f"[WARN] No data fetched for token {token}")
                return None
                
            # Convert to Polars DataFrame
            # Angel One format: ["2026-02-12T09:15...", O, H, L, C, V]
            schema = {"timestamp": pl.Utf8, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64}
            
            # Polars native construction from list of lists is fast
            data = response['data']
            df = pl.DataFrame(data, schema=schema, orient="row")
            
            # Parse Timestamp with timezone handling
            # Strip timezone if present (e.g. +05:30) to avoid parsing errors
            try:
                df = df.with_columns(
                    pl.col("timestamp")
                    .str.replace(r"\+.*$", "") # Remove +05:30
                    .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S")
                )
            except Exception as e:
                 logger.warning(f"Timestamp Parse Fallback: {e}")
                 # Fallback
                 try:
                     df = df.with_columns(pl.col("timestamp").str.to_datetime())
                 except:
                     pass

            return df

        except Exception as e:
            logger.error(f"[ERROR] API Fetch Error: {e}")
            return None

    def update_parquet(self, ticker, token):
        """
        The Stitching Logic: Loads Parquet -> Fetches Delta -> Merges -> Saves.
        """
        file_path = f"{self.ohlcv_path}/{ticker}.parquet"
        logger.info(f"[INFO] Updating {ticker}...")

        # 1. Fetch New Data (The Delta)
        # Fetch enough history to cover gaps (e.g., 30 days)
        new_df = self.fetch_recent_history(token, days=30)
        
        # 2. Check if we have data to process
        if new_df is None: return

        final_df = None

        # 3. Load Old Data (The History)
        if os.path.exists(file_path):
            try:
                old_df = pl.read_parquet(file_path)

                # SELF-HEALING: Check for Truncated History
                if len(old_df) < 200:
                    logger.warning(f"[WARN] {ticker} has distinct lack of history ({len(old_df)} rows). Forcing full backfill...")
                    raise ValueError("History too short")

                # The Stitch (Anti-Join + Concat)
                if not old_df.is_empty():
                    last_time = old_df["timestamp"].max()
                    new_df_filtered = new_df.filter(pl.col("timestamp") > last_time)
                else:
                    new_df_filtered = new_df
                
                if new_df_filtered.is_empty():
                    logger.info(f"   -> {ticker} is already up to date.")
                    return

                final_df = pl.concat([old_df, new_df_filtered])
                logger.info(f"   -> Appended {len(new_df_filtered)} new candles.")
            except Exception as e:
                logger.error(f"[ERROR] Error reading existing parquet {file_path}: {e}")
                # Backup and overwrite (Force replace on Windows)
                if os.path.exists(file_path + ".bak"):
                    os.remove(file_path + ".bak")
                os.rename(file_path, file_path + ".bak")
                
                # RE-FETCH FULL HISTORY (2000 Days) to avoid truncation
                logger.warning(f"[WARN] File corrupted/incompatible. Fetching full history (2000d) for {ticker}...")
                time.sleep(1) # Pause for rate limit
                full_df = self.fetch_recent_history(token, days=2000)
                final_df = full_df if full_df is not None else new_df
            
        else:
            # First time initialization -> Need FULL HISTORY
            logger.info(f"   -> New Database Created for {ticker} (Fetching 2000 days)")
            # Wait a bit longer for heavy fetch
            time.sleep(1)
            full_df = self.fetch_recent_history(token, days=2000)
            final_df = full_df if full_df is not None else new_df

        if final_df is not None and not final_df.is_empty():
            # 4. Deduplication (Safety Net)
            final_df = final_df.unique(subset=["timestamp"]).sort("timestamp")
            
            # 5. Atomic Write (Save)
            final_df.write_parquet(file_path)
            last_date = final_df["timestamp"].max()
            logger.info(f"[OK] {ticker} Saved. Rows: {len(final_df)} | Latest: {last_date}")
            print(f"   -> Latest Data: {last_date}")

    def record_depth_snapshot(self, ticker, depth_data):
        """
        Appends a snapshot of Market Depth to Parquet.
        depth_data: dict with 'ltp', 'friction', 'total_buy', 'total_sell', 'bids', 'asks', 'timestamp'
        """
        file_path = f"{self.depth_path}/{ticker}_depth.parquet"
        
        # Flatten dict to row
        row = {
            "timestamp": depth_data.get('timestamp', datetime.now()),
            "ltp": depth_data.get('ltp', 0.0),
            "friction": depth_data.get('friction', 0.0),
            "buy_pressure": depth_data.get('total_buy_pressure', 0),
            "sell_pressure": depth_data.get('total_sell_pressure', 0)
        }
        
        new_row_df = pl.DataFrame([row])
        
        if os.path.exists(file_path):
            try:
                # Append mode via reading and writing 
                old_df = pl.read_parquet(file_path)
                final_df = pl.concat([old_df, new_row_df])
                final_df.write_parquet(file_path)
            except:
                new_row_df.write_parquet(file_path)
        else:
             new_row_df.write_parquet(file_path)

if __name__ == "__main__":
    foundry = ChimeraFoundry()
    
    # 1. Load All Mapped Tickers
    all_tickers = list(foundry.token_map.keys()) # ['RELIANCE', 'TCS', ...]
    
    # 2. Prioritize Active Strategy Tickers
    active_tickers = [t.replace('.NS', '') for t in CONFIG.get('TICKERS', [])]
    
    # Split into Active vs Rest
    priority_list = [t for t in all_tickers if t in active_tickers]
    rest_list = [t for t in all_tickers if t not in active_tickers]
    
    # Combine (Priority First)
    final_list = priority_list + rest_list
    
    print(f"[INFO] Foundry Target: {len(final_list)} tickers (Priority: {len(priority_list)} active).")
    
    # 3. Update Loop
    print(f"[INFO] Starting Lake Sync... (Cntrl+C to stop)")
    logger.info(f"Starting Lake Sync for {len(final_list)} tickers")
    
    for i, ticker_base in enumerate(final_list):
        # foundry.token_map keys are "RELIANCE", "TCS"
        # We need to pass the token directly or reconstruct full ticker for logging
        token_info = foundry.token_map.get(ticker_base)
        if not token_info: continue
            
        token = token_info['token']
        symbol_for_file = f"{ticker_base}.NS" # Reconstruct filename format
        
        print(f"[{i+1}/{len(all_tickers)}] Syncing {symbol_for_file}...")
        foundry.update_parquet(symbol_for_file, token)
