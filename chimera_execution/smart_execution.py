import asyncio
import random
from config import *

# --------------------------------------------------
# EXECUTION THROTTLER
# --------------------------------------------------

async def throttle():
    """
    Randomized delay to reduce execution footprint and avoid rate limits.
    """
    delay = random.uniform(THROTTLE_MIN_DELAY, THROTTLE_MAX_DELAY)
    # print(f"⏳ Throttling for {delay:.2f}s...") 
    await asyncio.sleep(delay)


# --------------------------------------------------
# VWAP STYLE SLICER
# --------------------------------------------------

def slice_order(total_qty):
    """
    Splits a large order into smaller chunks to minimize market impact.
    """
    slices = []
    remaining = total_qty

    while remaining > 0:
        # Determine chunk size (percent of total, but ensure at least 1 qty)
        chunk_size = int(total_qty * MAX_ORDER_CHUNK)
        chunk = max(1, chunk_size)

        qty = min(chunk, remaining)
        slices.append(qty)

        remaining -= qty

    return slices


# --------------------------------------------------
# LIQUIDITY SCANNER (SIMPLE VERSION)
# --------------------------------------------------

def liquidity_adjust(weight, volatility):
    """
    Adjusts order weight based on market volatility.
    Instead of blocking trades, we scale them down during stress.
    """
    if volatility > VOL_SPIKE_THRESHOLD:
        print(f"⚠️ High Volatility ({volatility:.2%}) > Threshold ({VOL_SPIKE_THRESHOLD:.2%}) — Scaling order by 40%")
        return weight * 0.6

    return weight


# --------------------------------------------------
# ORDER RETRY LOGIC
# --------------------------------------------------

async def safe_place_order(broker, params):
    """
    Robust order placement with retry logic.
    """
    for i in range(RETRY_LIMIT):
        try:
            # Check if broker is connected
            if not broker or not broker.api:
                print("❌ Broker not connected. Simulating success for testing.")
                return "SIMULATED_ID_123"

            orderId = await broker.place_order(params)
            
            if orderId:
                print(f"✅ Order Success: {params['transactiontype']} {params['quantity']} {params['tradingsymbol']} | ID: {orderId}")
                return orderId
            else:
                raise Exception("Broker returned None for Order ID")

        except Exception as e:
            print(f"⚠️ Retry {i+1}/{RETRY_LIMIT} Failed for {params['tradingsymbol']}: {e}")
            await asyncio.sleep(1)

    print(f"❌ Order Failed Permanently for {params['tradingsymbol']}")
    return None
