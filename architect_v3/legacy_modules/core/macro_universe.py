class MacroScanner:
    def __init__(self):
        # Critical Indices and Commodities for Indian Markets
        self.indices = {
            "^NSEI": "Nifty 50",
            "^NSEBANK": "Bank Nifty",
            "^CNXIT": "Nifty IT",
            "^INDIAVIX": "India VIX"  # CRITICAL for Risk Management
        }
        
        # Commodity / Liquid ETFs (The "Cash" Proxies)
        self.etfs = {
            "GOLDBEES.NS": "Gold ETF",
            "SILVERBEES.NS": "Silver ETF",
            "LIQUIDBEES.NS": "Cash Equivalent"
        }

    def get_macro_tickers(self):
        """Returns a list of all macro tickers to ingest."""
        tickers = list(self.indices.keys()) + list(self.etfs.keys())
        return tickers

# --- SELF-LEARNING AUDIT ---
# Logic Check: Why use tickers starting with '^'? 
# Answer: Yahoo uses '^' for Indices. They are not tradable assets (you can't buy the index directly), 
# but they are essential for calculating signals.
