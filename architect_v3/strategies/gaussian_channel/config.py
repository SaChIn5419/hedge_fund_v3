# strategies/architect/config.py

# DHAN API CREDENTIALS
DHAN_CLIENT_ID = "YOUR_CLIENT_ID"
DHAN_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"

# MICRO-CAP RISK PARAMETERS (5 Lakh Base)
TOTAL_CAPITAL = 500000
MAX_POSITIONS = 5
CAPITAL_PER_POSITION = TOTAL_CAPITAL / MAX_POSITIONS # ₹1,00,000 per stock

# TIMEFRAME
TIMEFRAME = '1D' # Shifted to Daily to eliminate STT drag

# REGIME THRESHOLDS
STOP_LOSS_ATR_MULT = 2.0 # Wider stop for daily volatility

# Local Data Source
PARQUET_PATH = "data/raw/*.parquet"
NIFTY_TICKER = "^NSEI" # Mapping ID 13 to Ticker
