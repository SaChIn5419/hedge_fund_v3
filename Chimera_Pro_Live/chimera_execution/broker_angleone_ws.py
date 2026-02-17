import time
import json
import threading
import random
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import pyotp
from .config import *  # Import config from the same package

class AngleOneWS:
    def __init__(self):
        self.mock_mode = getattr(globals().get('config', {}), 'MOCK_MODE', False)
        # Re-import config to be sure
        from . import config
        self.mock_mode = config.MOCK_MODE

        if not self.mock_mode:
            try:
                self.api = SmartConnect(api_key=API_KEY)
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize SmartConnect: {e}")
                self.api = None
        else:
            print("⚠️ RUNNING IN MOCK MODE (No Real Broker Connection)")
            self.api = "MOCK_API"
            
        self.feedToken = None
        self.ws = None
        self.is_connected = False
        self.mock_running = False

    # --------------------------------------------------
    # LOGIN
    # --------------------------------------------------
    def login(self):
        if self.mock_mode:
            print("✅ [MOCK] AngleOne WS Authenticated")
            return True

        if not self.api:
            print("❌ API not initialized.")
            return False

        try:
            totp = pyotp.TOTP(TOTP_SECRET).now()
            data = self.api.generateSession(CLIENT_ID, PASSWORD, totp)

            if not data["status"]:
                raise Exception(f"Login failed: {data['message']}")

            self.feedToken = self.api.getfeedToken()
            self.jwtToken = data["data"]["jwtToken"] # Capture JWT Token
            print("✅ AngleOne WS Authenticated")
            return True
            
        except Exception as e:
            print(f"❌ Login Error: {e}")
            return False

    # --------------------------------------------------
    # START WEBSOCKET STREAM
    # --------------------------------------------------
    def start(self, token_list, on_tick_callback):
        """
        Starts the websocket connection and stream.
        """
        if self.mock_mode:
            self.start_mock_simulation(token_list, on_tick_callback)
            return

        if not self.api or not self.feedToken:
            print("❌ Cannot start WS: Not logged in.")
            return

        self.ws = SmartWebSocketV2(
            auth_token=self.jwtToken, 
            api_key=API_KEY,
            client_code=CLIENT_ID,
            feed_token=self.feedToken
        )

        def on_data(wsapp, message):
            try:
                if isinstance(message, str):
                    message = json.loads(message)
                
                tick = self.parse_tick(message)
                if tick:
                    on_tick_callback(tick)
            except Exception as e:
                pass

        def on_open(wsapp):
            print("✅ Websocket Connection Opened")
            self.is_connected = True
            self.ws.subscribe(
                correlation_id="chimera_stream",
                mode=1, 
                token_list=token_list
            )
            print(f"📡 Subscribed to {len(token_list)} exchange segments")

        def on_error(wsapp, error):
            print(f"❌ Websocket Error: {error}")

        def on_close(wsapp):
            print("⚠️ Websocket Connection Closed")
            self.is_connected = False

        self.ws.on_data = on_data
        self.ws.on_open = on_open
        self.ws.on_error = on_error
        self.ws.on_close = on_close

        self.ws.connect()

    def start_mock_simulation(self, token_list, callback):
        """
        Simulates live ticks for testing.
        """
        print("🎭 Starting Mock Simulation Loop...")
        self.mock_running = True
        
        # Extract tokens from the list structure
        # param token_list is usually [{"exchangeType": 1, "tokens": ["2885"]}]
        tokens = []
        for item in token_list:
            tokens.extend(item["tokens"])
            
        # Initial prices
        prices = {t: 1000.0 + random.uniform(-10, 10) for t in tokens}
        
        def run_sim():
            while self.mock_running:
                for t in tokens:
                    # Random Walk
                    change = random.uniform(-0.5, 0.5)
                    prices[t] += change
                    
                    tick = {
                        "symboltoken": t,
                        "price": prices[t],
                        "timestamp": time.time(),
                        "vol_traded": random.randint(100, 50000)
                    }
                    callback(tick)
                    time.sleep(0.1) # Fast ticks
                    
        t = threading.Thread(target=run_sim, daemon=True)
        t.start()

    # --------------------------------------------------
    # SAFE JSON PARSE
    # --------------------------------------------------
    def parse_tick(self, msg):
        try:
            if "last_traded_price" in msg:
                return {
                    "symboltoken": msg.get("token"),
                    "price": float(msg.get("last_traded_price")) / 100.0, 
                    "timestamp": msg.get("exchange_timestamp"),
                    "vol_traded": msg.get("volume_trade_for_the_day", 0)
                }
            return None
        except:
            return None
