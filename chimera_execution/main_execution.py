import asyncio
import pandas as pd
import sys
import os

# Add parent directories to path to import chimera_protocol modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from chimera_execution.config import *
from chimera_execution.broker_angleone_async import AsyncAngleOneBroker
from chimera_execution.async_execution_engine import execute_signals
from chimera_execution.signal_generator import ChimeraSignalGenerator
from chimera_protocol.chimera_live import ChimeraLiveProtocol

async def run_batch_execution():
    print("==========================================")
    print("   CHIMERA SMART EXECUTION ENGINE v2.0    ")
    print("   (Dynamic Signal Generation Mode)       ")
    print("==========================================")

    # 1. Initialize Broker
    broker = AsyncAngleOneBroker()
    if not broker.login():
        print("❌ Aborting Execution: Broker Login Failed")
        return

    # 2. Generate Signals LIVE
    print("\n📡 Generating Live Signals from Signal Engine...")
    sig_gen = ChimeraSignalGenerator()
    latest_signals = sig_gen.generate_signals()
    
    if latest_signals.empty:
        print("ℹ️ No actionable signals generated for today (or Defensive Mode active).")
        return

    # 3. Check Risk Protocol (Chimera Live)
    # Note: We need a historical log for Chimera Live Protocol to work fully.
    # For now, we assume the Signal Generator has handled the primary Regime/Risk checks.
    # The 'ChimeraLiveProtocol' was designed to audit a backtest log. 
    # Since we are generating fresh signals, the SignalGenerator includes the Layer 1 checks.
    # Layer 2 & 3 checks (Execution Reality, Survival) rely on account history which we might fetch from broker later.
    
    print("\n✅ Signals Generated:")
    print(latest_signals[['tradingsymbol', 'weight', 'structure']])

    # 4. Prepare for Execution
    # Ensure columns match requirements (tradingsymbol, symboltoken, weight, volatility)
    # Signal Generator already provides these except symboltoken needs mapping logic
    
    # Placeholder for Token Mapper (In production, replace with actual API call or Master JSON)
    latest_signals['symboltoken'] = "0000" 

    # 5. Execute
    await execute_signals(broker, latest_signals)

# ==========================================
# ⚡ PRO UPGRADE: WEBSOCKET EVENT LOOP
# ==========================================

async def run_websocket_execution():
    print("\n📡 Starting WebSocket Event Loop (Not Implemented in Basic Mode)...")
    # await websocket_engine.start()

if __name__ == "__main__":
    try:
        asyncio.run(run_batch_execution())
    except KeyboardInterrupt:
        print("\n🛑 Execution Stopped by User")
