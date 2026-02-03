import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class AsynchronousCircuitBreaker:
    """
    THE WATCHDOG DAEMON.
    """
    def __init__(self, 
                 market_ticker='^CRSLDX',  # CRISIL DEX
                 vol_ticker='^INDIAVIX', # India VIX
                 sma_window=200, 
                 crash_threshold=-0.03): # -3% Daily Drop trigger
        
        self.market_ticker = market_ticker
        self.vol_ticker = vol_ticker
        self.sma_window = sma_window
        self.crash_threshold = crash_threshold
        
        # Hysteresis State: 'NORMAL' or 'DEFENSIVE'
        self.current_state = 'NORMAL' 

    def fetch_vital_signs(self):
        """
        Fetches the latest heartbeat of the market.
        """
        # Get data for the last year to calc SMA200
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)
        
        print(f"--- DAEMON SCANNING: {end_date.date()} ---")
        
        # Fetch Market Data
        # auto_adjust=True ensures we get clean price data
        market = yf.download(self.market_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if market.empty:
            print(f"[ERROR] Daemon blinded. Cannot fetch Market Data ({self.market_ticker}).")
            # Try fallback to Nifty 500 if Nifty 50 fails? No, user specified Nifty 50 logic.
            return None

        # Clean MultiIndex if present (yfinance v0.2+)
        if isinstance(market.columns, pd.MultiIndex):
            # If 'Close' is in the columns
            try:
                market = market.xs('Close', axis=1, level=0) if 'Close' in market.columns.get_level_values(0) else market
            except:
                pass # structure might be simple
        elif 'Close' in market.columns:
            market = market[['Close']]

        # Fetch Volatility Data
        try:
            vix = yf.download(self.vol_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not vix.empty:
                if isinstance(vix.columns, pd.MultiIndex):
                     try:
                        vix = vix.xs('Close', axis=1, level=0) if 'Close' in vix.columns.get_level_values(0) else vix
                     except:
                        pass
                
                # Take the last available VIX close
                # Use iloc[-1] from the single column Series or DataFrame
                if isinstance(vix, pd.DataFrame) and 'Close' in vix.columns:
                    current_vix = vix['Close'].iloc[-1]
                else:
                    current_vix = vix.iloc[-1]
                    if isinstance(current_vix, pd.Series): # If mistakenly still a row
                        current_vix = current_vix.item()
            else:
                current_vix = 0.0
        except Exception as e:
            print(f"[WARN] VIX Fetch failed: {e}")
            current_vix = 0.0 # Default/Ignore
            
        return market, current_vix

    def analyze_physics(self, market_df, current_vix):
        """
        Determines if the 'Building is on Fire'.
        """
        # Ensure we work with a Series for rolling
        if isinstance(market_df, pd.DataFrame):
            if 'Close' in market_df.columns:
                close_series = market_df['Close']
            else:
                # Assume single column DF
                close_series = market_df.iloc[:, 0]
        else:
            close_series = market_df

        # 2. CALCULATE INDICATORS
        # The `market_df` passed to this function is already a DataFrame with a 'Close' column
        # or a Series. We need to ensure `history` refers to this.
        history = market_df if isinstance(market_df, pd.DataFrame) else pd.DataFrame(market_df, columns=['Close'])

        current_price = history['Close'].iloc[-1]
        sma_200 = history['Close'].rolling(window=self.sma_window).mean().iloc[-1] # Use self.sma_window
        daily_drop = history['Close'].pct_change().iloc[-1]
        
        # 3. GET VIX (FEAR GAUGE) - current_vix is already passed in
        vix = current_vix # Use the VIX already fetched by fetch_vital_signs

        print(f"NIFTY LEVEL:   {current_price:.2f}")
        if self.current_state == 'NORMAL':
            # Trigger: Daily drop exceeds threshold (e.g. -3%)
            if daily_drop < self.crash_threshold:
                print(f"[ALERT] CRASH DETECTED. Drop: {daily_drop:.2%}")
                self.current_state = 'DEFENSIVE'
                return "LIQUIDATE"
            
        elif self.current_state == 'DEFENSIVE':
            # TO RESET TO NORMAL: Needs Price > SMA200 (Strong recovery) OR VIX < 18 (Calm)
            if (current_price > sma_200) or (current_vix < 18.0):
                print(">>> STORM PASSED. RELEASING CIRCUIT BREAKER. <<<")
                self.current_state = 'NORMAL'
                return "RESUME"
        
        print(f"SYSTEM STATE:  {self.current_state}")        
        return "HOLD"

    def execute(self):
        """
        The Main Loop. Run this every day at 3:15 PM.
        """
        data = self.fetch_vital_signs()
        if data is None: return "ERROR"
        
        market, vix = data
        action = self.analyze_physics(market, vix)
        
        print(f"DAEMON VERDICT: [{action}]")
        return action

if __name__ == "__main__":
    # 1. Initialize Once
    watchdog = AsynchronousCircuitBreaker()

    # 2. Run Daily
    today_action = watchdog.execute()

    # 3. Connect to your Order Management System (OMS)
    if today_action == "LIQUIDATE":
        print("SENDING API CALL: SELL ALL POSITIONS -> BUY LIQUIDBEES")
        # your_broker_api.place_order(...) 
        
    elif today_action == "RESUME":
        print("SENDING API CALL: RE-ACTIVATE WEEKLY STRATEGY")
        # your_broker_api.buy_back(...)
