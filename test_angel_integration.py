import sys
import os
import asyncio

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chimera_execution.broker_angleone_async import AsyncAngleOneBroker
from chimera_protocol.chimera_engine import validate_vacuum_with_depth

async def test_integration():
    print("Testing Angel One Integration Components...")
    
    # 1. Test Broker Instantiation and Token Map
    try:
        broker = AsyncAngleOneBroker()
        print("[OK] Broker Instantiated")
        
        token = broker.get_token("RELIANCE.NS")
        if token:
            print(f"[OK] Token Map Working: RELIANCE.NS -> {token}")
        else:
            print("[WARN] Token Map returned None (Ensure map file exists)")
            
    except Exception as e:
        print(f"[FAIL] Broker Test Failed: {e}")

    # 2. Test Physics Logic
    try:
        # Case A: Good Vacuum
        mock_data_good = {'friction': 1.5, 'ltp': 100}
        res_good = validate_vacuum_with_depth("BULL_TURBO", mock_data_good)
        if res_good:
            print("[OK] Physics Check (Good) Passed")
        else:
            print("[FAIL] Physics Check (Good) Failed")
            
        # Case B: Bad Vacuum (Wall)
        mock_data_bad = {'friction': 4.0, 'ltp': 100}
        res_bad = validate_vacuum_with_depth("BULL_TURBO", mock_data_bad)
        if not res_bad:
            print("[OK] Physics Check (Bad) Blocked Correctly")
        else:
            print("[FAIL] Physics Check (Bad) Failed to Block")
            
    except Exception as e:
         print(f"[FAIL] Physics Test Failed: {e}")

    print("Test Complete.")

if __name__ == "__main__":
    asyncio.run(test_integration())
