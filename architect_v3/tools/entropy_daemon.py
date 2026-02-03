import pandas as pd
import numpy as np

class EntropyDaemon:
    """
    PHASE 7 ENGINE: RECONCILIATION
    Closes the loop between Algo Execution and Broker Settlement.
    """
    def __init__(self, tolerance_price=0.05, tolerance_time_sec=60):
        self.epsilon_price = tolerance_price
        self.epsilon_time = pd.Timedelta(seconds=tolerance_time_sec)

    def ingest_ledgers(self, internal_path, broker_path):
        """
        Loads the two conflicting realities.
        """
        try:
            # 1. Internal Log (The Algo's Memory)
            self.internal = pd.read_csv(internal_path)
            self.internal['timestamp'] = pd.to_datetime(self.internal['timestamp'])
            
            # 2. Broker Report (The Street's Truth)
            self.external = pd.read_csv(broker_path)
            self.external['timestamp'] = pd.to_datetime(self.external['timestamp'])
        except Exception as e:
            print(f"CRITICAL: Ledger Ingestion Failed. {e}")
            self.internal = pd.DataFrame()
            self.external = pd.DataFrame()

    def detect_trade_breaks(self):
        """
        Matches trades based on Symbol, Quantity, and Approximate Time.
        Identifies 'Phantom Fills' and 'Missed Trades'.
        """
        print("--- RUNNING TRADE RECONCILIATION ---")
        
        if self.internal.empty or self.external.empty:
            print("WARNING: Ledgers are empty. Skipping reconciliation.")
            return pd.DataFrame()

        # Create a copy to track matched trades
        street_ledger = self.external.copy()
        breaks = []

        for idx, algo_trade in self.internal.iterrows():
            # Filter Street Ledger for potential matches
            # Criteria: Same Symbol, Same Side, Time within window
            # Note: Assuming 'side' is consistently uppercase (BUY/SELL)
            candidates = street_ledger[
                (street_ledger['symbol'] == algo_trade['symbol']) & 
                (street_ledger['side'] == algo_trade['side']) & 
                (street_ledger['quantity'] == algo_trade['quantity']) &
                (abs(street_ledger['timestamp'] - algo_trade['timestamp']) <= self.epsilon_time)
            ]

            if candidates.empty:
                # TYPE 1 BREAK: PHANTOM FILL
                # Algo thinks it traded. Broker says no.
                breaks.append({
                    'type': 'PHANTOM_FILL',
                    'symbol': algo_trade['symbol'],
                    'qty': algo_trade['quantity'],
                    'algo_price': algo_trade['price'],
                    'broker_price': np.nan,
                    'delta': np.nan
                })
            else:
                # Check Price Discrepancy (Slippage/Fees)
                match = candidates.iloc[0] # Take best match
                price_delta = abs(algo_trade['price'] - match['price'])
                
                if price_delta > self.epsilon_price:
                    # TYPE 2 BREAK: PRICE MISMATCH (Slippage Exceeded)
                    breaks.append({
                        'type': 'PRICE_MISMATCH',
                        'symbol': algo_trade['symbol'],
                        'qty': algo_trade['quantity'],
                        'algo_price': algo_trade['price'],
                        'broker_price': match['price'],
                        'delta': price_delta
                    })
                
                # Remove matched trade from street ledger to prevent double counting
                street_ledger = street_ledger.drop(match.name)

        # TYPE 3 BREAK: UNRECORDED TRADE
        # Broker has a trade. Algo has no memory of it.
        # (Remaining rows in street_ledger)
        for idx, street_trade in street_ledger.iterrows():
             breaks.append({
                    'type': 'UNRECORDED_TRADE',
                    'symbol': street_trade['symbol'],
                    'qty': street_trade['quantity'],
                    'algo_price': np.nan,
                    'broker_price': street_trade['price'],
                    'delta': np.nan
                })
        
        return pd.DataFrame(breaks)

    def reconcile_cash_and_friction(self):
        """
        Audit the entropy (Fees).
        """
        if self.internal.empty or self.external.empty:
            return

        algo_fees = self.internal['commission'].sum()
        broker_fees = self.external['commission'].sum()
        
        delta = algo_fees - broker_fees
        
        print(f"--- FRICTION AUDIT ---")
        print(f"Projected Friction: INR {algo_fees:,.2f}")
        print(f"Realized Friction:  INR {broker_fees:,.2f}")
        print(f"Leakage (Delta):    INR {delta:,.2f}")
        
        if delta < 0:
            print("[WARNING] Broker charging more than modeled.")

if __name__ == "__main__":
    # Test Run
    internal_data = {
        'timestamp': ['2023-10-27 10:00:00', '2023-10-27 10:05:00'],
        'symbol': ['RELIANCE', 'TCS'],
        'side': ['BUY', 'SELL'],
        'quantity': [100, 50],
        'price': [2400.00, 3500.00],
        'commission': [50.0, 30.0]
    }
    external_data = {
        'timestamp': ['2023-10-27 10:00:02', '2023-10-27 10:05:01'], # Time drift
        'symbol': ['RELIANCE', 'TCS'],
        'side': ['BUY', 'SELL'],
        'quantity': [100, 50],
        'price': [2401.50, 3500.00], # Slippage on Reliance
        'commission': [55.0, 30.0]  # Higher fee on Reliance
    }

    pd.DataFrame(internal_data).to_csv('internal_log_test.csv', index=False)
    pd.DataFrame(external_data).to_csv('broker_report_test.csv', index=False)

    daemon = EntropyDaemon(tolerance_price=1.0)
    daemon.ingest_ledgers('internal_log_test.csv', 'broker_report_test.csv')
    df = daemon.detect_trade_breaks()
    print(df)
    daemon.reconcile_cash_and_friction()
