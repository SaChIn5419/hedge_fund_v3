import sys
import os
import pandas as pd
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.entropy_daemon import EntropyDaemon

def run_chaos_monkey():
    print("--- CHAOS MONKEY: TESTING ENTROPY DAEMON ---")
    
    # 1. Generate PERFECT Ledger (Baseline)
    base_data = {
        'timestamp': [pd.Timestamp('2023-01-01 10:00:00'), pd.Timestamp('2023-01-01 11:00:00'), pd.Timestamp('2023-01-01 12:00:00')],
        'symbol': ['RELIANCE', 'TCS', 'INFY'],
        'side': ['BUY', 'SELL', 'BUY'],
        'quantity': [100, 50, 200],
        'price': [2500.0, 3200.0, 1500.0],
        'commission': [10.0, 10.0, 10.0]
    }
    internal = pd.DataFrame(base_data)
    external = internal.copy()
    
    # 2. Inject CHAOS (Errors)
    print("Injecting Errors...")
    
    # ERROR A: PHANTOM FILL (Internal has record, Broker does not)
    # We delete TCS from External
    external = external[external['symbol'] != 'TCS']
    print("- Deleted 'TCS' from Broker Report (Simulating Phantom Fill)")
    
    # ERROR B: SLIPPAGE (Price Mismatch)
    # Reliance Price: 2500 -> 2505 (0.2% deviation > epsilon)
    external.loc[external['symbol'] == 'RELIANCE', 'price'] = 2505.0
    print("- Slipping 'RELIANCE' price by 5 INR")

    # ERROR C: UNRECORDED TRADE (Broker has trade, Algo does not)
    # Add HDFC to External
    new_trade = pd.DataFrame([{
        'timestamp': pd.Timestamp('2023-01-01 13:00:00'),
        'symbol': 'HDFC', 'side': 'BUY', 'quantity': 100, 'price': 1600.0, 'commission': 10.0
    }])
    external = pd.concat([external, new_trade], ignore_index=True)
    print("- Added 'HDFC' to Broker Report (Simulating Rogue Trade)")

    # 3. Save Mock Files
    internal.to_csv('mock_internal.csv', index=False)
    external.to_csv('mock_external.csv', index=False)
    
    # 4. Run Daemon
    print("\n--- DAEMON ACTIVATION ---")
    daemon = EntropyDaemon(tolerance_price=1.0)
    daemon.ingest_ledgers('mock_internal.csv', 'mock_external.csv')
    breaks = daemon.detect_trade_breaks()
    
    print("\n--- RESULTS ---")
    print(breaks)
    
    # 5. Assertions
    failures = []
    
    # Check Phantom Fill
    if not ((breaks['type'] == 'PHANTOM_FILL') & (breaks['symbol'] == 'TCS')).any():
        failures.append("FAILED to detect Phantom Fill (TCS)")
        
    # Check Price Mismatch
    if not ((breaks['type'] == 'PRICE_MISMATCH') & (breaks['symbol'] == 'RELIANCE')).any():
        failures.append("FAILED to detect Price Mismatch (RELIANCE)")
        
    # Check Unrecorded
    if not ((breaks['type'] == 'UNRECORDED_TRADE') & (breaks['symbol'] == 'HDFC')).any():
        failures.append("FAILED to detect Unrecorded Trade (HDFC)")
        
    if not failures:
        print("\n>>> TEST PASSED: DAEMON IS 100% VIGILANT <<<")
    else:
        print("\n>>> TEST FAILED <<<")
        for f in failures: print(f"[x] {f}")

if __name__ == "__main__":
    run_chaos_monkey()
