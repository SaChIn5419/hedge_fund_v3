import time
import random
import asyncio
from SmartApi import SmartConnect
import pyotp
import os
import sys
import importlib.util

# Robust Credential Loading
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../Chimera_Pro_Live/chimera_execution
pro_live_path = os.path.abspath(os.path.join(current_dir, "..")) # .../Chimera_Pro_Live
root_path = os.path.abspath(os.path.join(pro_live_path, "..")) # .../hedge_fund_v3

try:
    # 1. Try importing from secrets (Local or Root)
    secrets_path = os.path.join(pro_live_path, "chimera_execution", "secrets.py")
    if not os.path.exists(secrets_path):
        secrets_path = os.path.join(root_path, "secrets.py")
    
    if os.path.exists(secrets_path):
        spec = importlib.util.spec_from_file_location("chimera_secrets_live", secrets_path)
        chimera_secrets = importlib.util.module_from_spec(spec)
        sys.modules["chimera_secrets_live"] = chimera_secrets
        spec.loader.exec_module(chimera_secrets)
        
        API_KEY = chimera_secrets.API_KEY
        CLIENT_ID = chimera_secrets.CLIENT_ID
        PASSWORD = chimera_secrets.PASSWORD
        TOTP_SECRET = chimera_secrets.TOTP_SECRET
        print(f"[OK] [Pro_Live] Loaded Credentials from {secrets_path}")
    else:
        raise ImportError("Secrets file not found")

except Exception as e:
    # 2. Fallback to config
    try:
        from .config import API_KEY, CLIENT_ID, PASSWORD, TOTP_SECRET
        print(f"[WARN] [Pro_Live] Loaded Credentials from config (Error: {e})")
    except ImportError:
        print("⚠️ Credentials not found.")

class AsyncAngleOneBroker:

    def __init__(self):
        self.mock_mode = getattr(globals().get('config', {}), 'MOCK_MODE', False)
        from . import config
        self.mock_mode = config.MOCK_MODE
        
        if not self.mock_mode:
            try:
                self.api = SmartConnect(api_key=API_KEY)
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize SmartConnect (Check dependencies): {e}")
                self.api = None
        else:
            self.api = "MOCK_API"

    def login(self):
        """
        Executes the Handshake Protocol with automated TOTP.
        """
        if self.mock_mode:
            print("✅ [MOCK] Async Broker Authenticated")
            return True

        if not self.api:
            print("❌ API not initialized.")
            return False

        try:
            print(f"🔐 Attempting Login for {CLIENT_ID}...")
            totp = pyotp.TOTP(TOTP_SECRET).now()
            data = self.api.generateSession(CLIENT_ID, PASSWORD, totp)

            if data['status']:
                # Capture session data for debugging/usage
                self.jwt_token = data['data']['jwtToken']
                self.refresh_token = data['data']['refreshToken']
                self.feed_token = self.api.getfeedToken()
                
                # OPTIONAL: Try to fetch profile, but DO NOT FAIL if it errors
                try:
                    profile = self.api.getProfile(self.refresh_token)
                    if not profile.get('status'):
                        print(f"⚠️ Warning: Could not fetch profile (Ignoring)")
                except Exception as e:
                     print(f"⚠️ Warning: Profile fetch error: {e}")

                print(f"✅ AngelOne Connected Successfully")
                return True  # ALWAYS Return True if Session is generated
            else:
                print(f"❌ Login Failed: {data['message']}")
                return False

        except Exception as e:
            print(f"❌ CRITICAL CONNECTION FAILURE: {e}")
            return False

    async def get_ltp(self, symbol, token):
        """
        Fetches Last Traded Price asynchronously.
        """
        if self.mock_mode:
            return 2500.0 + random.uniform(-10, 10)

        if not self.api: return 0.0

        loop = asyncio.get_event_loop()

        try:
            # Running blocking SDK call in executor
            resp = await loop.run_in_executor(
                None,
                lambda: self.api.ltpData(EXCHANGE, symbol, token)
            )
            
            if resp['status'] and resp['data']:
                return float(resp["data"]["ltp"])
            else:
                print(f"⚠️ LTP Fetch Failed for {symbol}")
                return 0.0
                
        except Exception as e:
            print(f"❌ Async LTP Error: {e}")
            return 0.0

    async def place_order(self, params):
        """
        Places an order asynchronously.
        """
        if self.mock_mode:
            order_id = f"MOCK_{random.randint(10000,99999)}"
            print(f"🛠️ [MOCK] Order Placed: {params['tradingsymbol']} {params['transactiontype']} Qty={params['quantity']} -> ID: {order_id}")
            return order_id
            
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
            print(f"❌ Place Order Error: {e}")
            raise e
