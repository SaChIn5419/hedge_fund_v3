import asyncio
import pandas as pd
import sys
import os

# Ensure we can import from neighbor directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from smart_execution import throttle, slice_order, liquidity_adjust, safe_place_order
from chimera_execution.broker_angleone_async import AsyncAngleOneBroker
from chimera_protocol.chimera_engine import validate_vacuum_with_depth
try:
    from chimera_protocol.data_foundry import ChimeraFoundry
    FOUNDRY = ChimeraFoundry() # Initialize Foundry for logging
    print("[INFO] Foundry Initialized for Depth Recording")
except ImportError:
    print("[WARN] Could not initialize Foundry. Depth data will not be saved.")
    FOUNDRY = None

async def execute_signals(broker, signal_df):
    """
    Executes a batch of signals using smart routing logic.
    
    Args:
        broker: Instance of AsyncAngleOneBroker
        signal_df: DataFrame with columns ['tradingsymbol', 'symboltoken', 'weight', 'volatility']
    """
    print(f"\n[START] Starting Batch Execution for {len(signal_df)} signals...")
    
    tasks = []

    for _, row in signal_df.iterrows():
        symbol = row.get('tradingsymbol') or row.get('ticker')
        token = str(row.get('symboltoken', ''))
        weight = row.get('weight', 0)
        volatility = row.get('volatility', 0)
        signal_type = row.get('signal_type', 'BULL_TURBO') # Default to TURBO if not specified

        # -------------------------------
        # 1) Fetch Microstructure (LTP + Depth)
        # -------------------------------
        # price = await broker.get_ltp(symbol, token) 
        # UPGRADE: Get Depth for Vacuum Check
        micro = await broker.get_market_microstructure(EXCHANGE, symbol, token)
        
        if not micro:
            print(f"[WARN] Skipping {symbol}: Microstructure Data Unavailable")
            continue
            
        price = micro['ltp']
        
        if price <= 0:
            print(f"[WARN] Skipping {symbol}: Invalid Price {price}")
            continue

        # -------------------------------
        # 1.2) RECORD DEPTH TO LAKE
        # -------------------------------
        if FOUNDRY:
            # Add timestamp execution time
            micro['timestamp'] = datetime.now()
            FOUNDRY.record_depth_snapshot(symbol, micro)

        # -------------------------------
        # 1.5) PHYSICS CHECK (Vacuum Validation)
        # -------------------------------
        if not validate_vacuum_with_depth(signal_type, micro):
            print(f"[BLOCK] BLOCKED {symbol}: Fake Vacuum Detected (High Friction)")
            continue

        # -------------------------------
        # 2) Liquidity adjustment
        # -------------------------------
        adj_weight = liquidity_adjust(weight, volatility)
        
        # Calculate capital allocation for this asset
        allocation = CAPITAL * LEVERAGE * abs(adj_weight)
        
        # Calculate Quantity
        qty = int(allocation / price)
        
        if qty <= 0:
            print(f"[WARN] Skipping {symbol}: Qty is 0 (Alloc: {allocation:.2f}, Price: {price})")
            continue

        side = "BUY" if adj_weight > 0 else "SELL"
        print(f"[TARGET] Strategy: {symbol} | Wgt: {weight:.2f} -> Adj: {adj_weight:.2f} | Qty: {qty} | Side: {side}")
        print(f"   (Friction: {micro['friction']:.2f} | Vacuum: Verified)")

        # -------------------------------
        # 3) Slice orders (VWAP style)
        # -------------------------------
        slices = slice_order(qty)
        print(f"[SLICE] Slicing {symbol} order into {len(slices)} chunks: {slices}")

        for i, q in enumerate(slices):
            
            params = {
                "variety": "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken": token,
                "transactiontype": side,
                "exchange": EXCHANGE,
                "ordertype": "MARKET",
                "producttype": "INTRADAY",
                "duration": "DAY",
                "price": "0",
                "quantity": str(q)
            }

            # Add throttling delay before dispatching
            await throttle()

            # Schedule the order
            tasks.append(
                safe_place_order(broker, params)
            )

    if not tasks:
        print("[INFO] No executable orders generated.")
        return []

    # Execute all order tasks concurrently
    print(f"[EXEC] Dispatching {len(tasks)} child orders to market...")
    results = await asyncio.gather(*tasks)

    print("[DONE] Batch Execution Complete")
    return results
