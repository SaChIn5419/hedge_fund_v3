# config.py

# CRYPTO RISK PARAMETERS (Assuming a $10,000 USD Account)
TOTAL_CAPITAL_USD = 10000.0
MAX_POSITIONS = 3
CAPITAL_PER_POSITION = TOTAL_CAPITAL_USD / MAX_POSITIONS # ~$3,333 per coin

# TIMEFRAME OVERRIDE
# In Equities, a 1D chart = 6.5 hours of trading.
# In Crypto, a 1D chart = 24 hours of trading. The Gaussian Channel reacts much faster here.
TIMEFRAME = '1D' 

# FRACTIONAL SHARES UNLOCKED
ALLOW_FRACTIONAL = True 

# BROKERAGE (Binance Spot)
MAKER_FEE = 0.0010 # 0.1%
TAKER_FEE = 0.0010 # 0.1%
SLIPPAGE = 0.0005  # 0.05%
