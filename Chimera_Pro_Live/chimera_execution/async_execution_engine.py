import asyncio
import pandas as pd
from .config import *  # Relative import fix
from .smart_execution import throttle, slice_order, liquidity_adjust, safe_place_order

async def execute_signals(broker, signal_df):
    """
    Executes a batch of signals using smart routing logic.
    
    Args:
        broker: Instance of AsyncAngleOneBroker
        signal_df: DataFrame with columns ['tradingsymbol', 'symboltoken', 'weight', 'volatility']
    """
    print(f"\n🚀 Starting Batch Execution for {len(signal_df)} signals...")
    
    tasks = []

    for _, row in signal_df.iterrows():
        symbol = row.get('tradingsymbol') or row.get('ticker')
        token = str(row.get('symboltoken', ''))
        weight = row.get('weight', 0)
        volatility = row.get('volatility', 0)

        # -------------------------------
        # 1) Fetch price
        # -------------------------------
        price = await broker.get_ltp(symbol, token)
        
        if price <= 0:
            print(f"⚠️ Skipping {symbol}: Invalid Price {price}")
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
            print(f"⚠️ Skipping {symbol}: Qty is 0 (Alloc: {allocation:.2f}, Price: {price})")
            continue

        side = "BUY" if adj_weight > 0 else "SELL"
        print(f"🎯 Strategy: {symbol} | Wgt: {weight:.2f} -> Adj: {adj_weight:.2f} | Qty: {qty} | Side: {side}")

        # -------------------------------
        # 3) Slice orders (VWAP style)
        # -------------------------------
        slices = slice_order(qty)
        print(f"🔪 Slicing {symbol} order into {len(slices)} chunks: {slices}")

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
        print("ℹ️ No executable orders generated.")
        return []

    # Execute all order tasks concurrently
    print(f"🔥 Dispatching {len(tasks)} child orders to market...")
    results = await asyncio.gather(*tasks)

    print("✅ Batch Execution Complete")
    return results
