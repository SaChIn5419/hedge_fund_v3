from SmartApi import SmartConnect
import pyotp
import asyncio
import json
import os
# Credentials Loading Strategy (Robust)
import sys
# Force add Chimera_Pro_Live to path to find secrets
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
pro_live_path = os.path.join(project_root, "Chimera_Pro_Live")

if pro_live_path not in sys.path:
    sys.path.append(pro_live_path)

import importlib.util
try:
    # 1. Try importing from secrets (Root or Chimera_Pro_Live) using Direct Path Loading
    # This avoids "secrets" module conflict and package ambiguity
    secrets_path = os.path.join(pro_live_path, "chimera_execution", "secrets.py")
    if not os.path.exists(secrets_path):
        # Checks root
        secrets_path = os.path.join(project_root, "secrets.py")
    
    if os.path.exists(secrets_path):
        spec = importlib.util.spec_from_file_location("chimera_secrets", secrets_path)
        chimera_secrets = importlib.util.module_from_spec(spec)
        sys.modules["chimera_secrets"] = chimera_secrets
        spec.loader.exec_module(chimera_secrets)
        
        API_KEY = chimera_secrets.API_KEY
        CLIENT_ID = chimera_secrets.CLIENT_ID
        PASSWORD = chimera_secrets.PASSWORD
        TOTP_SECRET = chimera_secrets.TOTP_SECRET
        print(f"[OK] Loaded Credentials from {secrets_path}")
    else:
        raise ImportError("Secrets file not found")

except Exception as e:
    # 2. Fallback to config
    try:
        from chimera_execution.config import API_KEY, CLIENT_ID, PASSWORD, TOTP_SECRET
        print(f"[WARN] Loaded Credentials from chimera_execution.config (Error was: {e})")
    except ImportError:
        print("⚠️ Credentials not found in secrets.py or config.py")

class AsyncAngleOneBroker:

    def __init__(self):
        try:
            self.api = SmartConnect(api_key=API_KEY)
            self.token_map = self.load_token_map()
        except Exception as e:
            print(f"[WARN] Warning: Could not initialize SmartConnect (Check dependencies): {e}")
            self.api = None
            self.token_map = {}

    def load_token_map(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "angel_token_map.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def get_token(self, symbol):
        # Symbol comes in as "RELIANCE.NS" or "RELIANCE"
        clean = symbol.replace(".NS", "").strip()
        if clean in self.token_map:
            return self.token_map[clean]['token']
        return None

    def login(self):
        if not self.api:
            print("[X] API not initialized.")
            return False
            
        try:
            print(f"[DEBUG] TOTP Secret Len: {len(TOTP_SECRET)}")
            totp_obj = pyotp.TOTP(TOTP_SECRET)
            totp = totp_obj.now()
            print(f"[DEBUG] Generated TOTP: {totp}")
            
            data = self.api.generateSession(CLIENT_ID, PASSWORD, totp)

            if not data["status"]:
                raise Exception(f"Login failed: {data['message']}")

            print("[OK] AngleOne Connected Successfully")
            return True
        except Exception as e:
            print(f"[X] Login Error: {e}")
            return False

    async def get_ltp(self, symbol, token=None):
        """
        Fetches Last Traded Price asynchronously.
        """
        if not self.api: return 0.0
        
        if token is None:
            token = self.get_token(symbol)
            if not token:
                print(f"[WARN] Token not found for {symbol}")
                return 0.0

        loop = asyncio.get_event_loop()

        try:
            # Running blocking SDK call in executor
            resp = await loop.run_in_executor(
                None,
                lambda: self.api.ltpData("NSE", symbol.replace(".NS", "-EQ"), token)
            )
            
            if resp['status'] and resp['data']:
                return float(resp["data"]["ltp"])
            else:
                # print(f"[WARN] LTP Fetch Failed for {symbol}")
                return 0.0
                
        except Exception as e:
            print(f"[X] Async LTP Error: {e}")
            return 0.0

    async def get_market_microstructure(self, exchange, symbol, token=None):
        """
        Fetches LTP and calculates Order Book Friction (Top 5).
        """
        if not self.api: return None
        
        if token is None:
            token = self.get_token(symbol)
            if not token: return None

        loop = asyncio.get_event_loop()
        try:
            # api.getMarketData(mode, exchangeTokens)
            # mode: "FULL", exchangeTokens: {"NSE": ["3045"]}
            tokens = {exchange: [token]}
            
            packet = await loop.run_in_executor(
                None, 
                lambda: self.api.getMarketData("FULL", tokens)
            )
            
            if not packet or not packet.get('status') or not packet.get('data'):
                return None
            
            # Data structure: {'fetched': [...], 'unfetched': [...]}
            fetched = packet['data'].get('fetched', [])
            if not fetched:
                return None
                
            data = fetched[0]
            
            # 1. Capture LTP
            ltp = float(data.get('ltp', 0.0))
            if ltp == 0.0: ltp = float(data.get('close', 0.0)) # Fallback
            
            # 2. Parse Depth
            depth = data.get('depth', {})
            buy_list = depth.get('buy', [])
            sell_list = depth.get('sell', [])
            
            buy_qty = sum([float(order.get('quantity', 0)) for order in buy_list])
            sell_qty = sum([float(order.get('quantity', 0)) for order in sell_list])
            
            # 3. Calculate Imbalance (The Physics Signal)
            friction_ratio = sell_qty / (buy_qty + 1e-9) # Avoid zero div
            
            return {
               "ltp": ltp,
               "friction": friction_ratio,
               "total_buy_pressure": buy_qty,
               "total_sell_pressure": sell_qty
            }

        except Exception as e:
            print(f"[WARN] Microstructure Data Failed for {symbol}: {e}")
            return None

    async def place_order(self, params):
        """
        Places an order asynchronously.
        """
        if not self.api: return None

        loop = asyncio.get_event_loop()

        try:
            # Running blocking SDK call in executor
            order_id = await loop.run_in_executor(
                None,
                lambda: self.api.placeOrder(params)
            )
            return order_id
        except Exception as e:
            print(f"[X] Place Order Error: {e}")
            raise e
