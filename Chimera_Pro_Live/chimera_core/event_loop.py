import asyncio
import json
import time
import os
from datetime import datetime
from chimera_execution.broker_angleone_ws import AngleOneWS
from chimera_execution.signal_generator import ChimeraSignalGenerator
from chimera_execution.async_execution_engine import execute_signals
from chimera_execution.broker_angleone_async import AsyncAngleOneBroker
from .state_machine import ExecutionStateMachine, STATE_IDLE, STATE_SIGNAL_READY, STATE_EXECUTING, STATE_COOLDOWN
from .physics_engine import ChimeraPhysicsEngine

# Governance Layers
from chimera_governance.drift_detector import RegimeDriftDetector
from chimera_governance.genome import StrategyGenome
from chimera_governance.capital_brain import CapitalAllocationBrain

class ChimeraEventLoop:
    def __init__(self):
        self.broker_ws = AngleOneWS()
        self.state_machine = ExecutionStateMachine()
        self.signal_generator = ChimeraSignalGenerator()
        self.async_broker = AsyncAngleOneBroker()
        self.physics_engine = ChimeraPhysicsEngine()
        
        # Initialize Governance
        self.drift_detector = RegimeDriftDetector()
        self.genome = StrategyGenome()
        self.capital_brain = CapitalAllocationBrain()
        
        self.market_state = {}
        self.neural_state_path = "data/neural_state.json"
        
        # Ensure data dir exists
        os.makedirs("data", exist_ok=True)
        
        # Load Evolution
        self.genome.load_dna()
        
        # Execution Queue
        self.pending_signals = []

    async def start(self):
        print("🚀 Starting Chimera Institutional Engine...")
        print(f"🧬 Active Genome: Lookback Energy={self.genome.get_gene('lookback_energy')}")
        
        # 1. Login to APIs
        if not self.broker_ws.login():
            print("❌ WS Login Failed")
            return
            
        print("⏳ Waiting 2s to respect API Rate Limits...")
        time.sleep(2)  # Prevent "Exceeding access rate" error
        self.async_broker.login()

        # 2. Subscribe to Tokens
        token_list = [
            {"exchangeType": 1, "tokens": ["2885", "1594", "3045"]} 
        ]
        
        print("📡 Connecting to Market Data Stream...")
        self.broker_ws.start(token_list, self.on_tick)
        
        # Keep the main loop alive and check for execution triggers
        while True:
            # Heartbeat (UI)
            self.update_neural_telemetry(None)
            
            # CHECK FOR EXECUTION
            if self.state_machine.current_state == STATE_SIGNAL_READY:
                print("⚡ EXECUTION TRIGGERED: State Machine is READY")
                
                # Create Signal Payload
                # In a real engine, we'd pull from a queue. For now, we construct from last physics.
                # We need to know WHICH token triggered it.
                # Ideally, handle_physics_signal should push to a queue.
                # For this hotfix, we check the market_state for the active trigger?
                # A better way: handle_physics_signal sets a 'pending_orders' list.
                
                if hasattr(self, 'pending_signals') and self.pending_signals:
                    print(f"🚀 Dispatching {len(self.pending_signals)} signals to Execution Engine...")
                    
                    # Execute
                    results = await execute_signals(self.async_broker, pd.DataFrame(self.pending_signals))
                    
                    if results:
                        print(f"✅ Execution Complete. Trades: {len(results)}")
                        self.state_machine.set_state(STATE_COOLDOWN)
                    else:
                        print("⚠️ Execution Failed or Empty")
                        self.state_machine.set_state(STATE_IDLE)
                    
                    self.pending_signals = [] # Clear
                else:
                    self.state_machine.set_state(STATE_IDLE)

            await asyncio.sleep(1)

    def on_tick(self, tick):
        """
        Triggered when a new tick arrives.
        """
        timestamp = datetime.now()
        token = tick.get('symboltoken')
        price = tick.get('price')
        
        # 1. Update Internal Market State
        self.market_state[token] = {
            "price": price,
            "time": timestamp
        }
        
        # 2. Governance Check (Drift & Capital)
        # We update Drift Detector with the price of the *Indices* ideally, 
        # but for now we update with the stock price to detect asset-level drift.
        # In production, we'd have a separate 'Index Stream' for regime.
        regime_data = self.drift_detector.on_tick(price)
        target_leverage = 0.0
        
        if regime_data != "INITIALIZING":
            target_leverage = self.capital_brain.update_allocation(regime_data)
        
        # 3. Feed Physics Engine
        physics_signal = self.physics_engine.on_tick(tick)
        
        # 4. Process Valid Physics Signal
        if physics_signal:
            print(f"⚛️ Physics Update {token}: E={physics_signal['energy']:.2f} | Leverage={target_leverage}x")
            self.handle_physics_signal(physics_signal, target_leverage)
        
        # 5. Update Neural State (Telemetry)
        self.update_neural_telemetry(tick, physics_signal, regime_data, target_leverage)

    def handle_physics_signal(self, signal, leverage):
        """
        Decides what to do with a new physics calculation.
        """
        if signal['energy'] > 0.2 and signal['structure'] == 'VACUUM':
            if self.state_machine.can_execute():
                print(f"🎯 TRIGGER: {signal['token']} Entering VACUUM. Leverage: {leverage}")
                
                # Construct Signal Payload for Execution Engine
                trade_signal = {
                    "token": signal['token'],
                    "symboltoken": signal['token'], # Mapping needed if different
                    "tradingsymbol": signal['token'], # Placeholder, needs proper mapping
                    "weight": 1.0, # Base weight, will be adjusted by leverage
                    "volatility": 0.015, # Proxy, should get from market_state
                    "signal_type": "BULL_TURBO",
                    "leverage": leverage
                }
                
                self.pending_signals.append(trade_signal)
                self.state_machine.set_state(STATE_SIGNAL_READY)

    def update_neural_telemetry(self, tick, physics_signal=None, regime_data="STABLE", leverage=1.0):
        """
        Writes the current system snapshot to JSON.
        """
        # Load existing state to preserve data if this is just a heartbeat
        existing_state = {}
        if not tick:
            try:
                with open(self.neural_state_path, "r") as f:
                    existing_state = json.load(f)
            except:
                pass
        
        # Determine values (New > Old > Default)
        current_tick = tick if tick else existing_state.get('latest_tick', {"symboltoken": "WAITING", "price": 0})
        current_regime = regime_data if regime_data != "INITIALIZING" else existing_state.get('regime', "INIT")
        current_leverage = leverage if leverage != 1.0 else existing_state.get('leverage', 1.0)
        current_physics = physics_signal if physics_signal else existing_state.get('last_physics', "No Change")

        state = {
            "timestamp": str(datetime.now()), # Always update this!
            "execution_state": self.state_machine.log_state(),
            "regime": current_regime,
            "leverage": current_leverage,
            "latest_tick": current_tick,
            "last_physics": current_physics,
            "market_depth": len(self.market_state)
        }
        
        try:
            with open(self.neural_state_path, "w") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
             pass

if __name__ == "__main__":
    loop = ChimeraEventLoop()
    asyncio.run(loop.start())
