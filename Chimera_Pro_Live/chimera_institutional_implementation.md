# Chimera Institutional Implementation Plan

## 🏰 Objective

Upgrade the isolated **Chimera Protocol** into the **"Conscious" Adaptive Trading Architecture** defined in the *Developer Codex*. This involves moving from a linear script to an event-driven, hierarchical organism with strict governance and risk management.

## ⚠️ Risk & Mitigation Strategy (The Audit)

*Derived from the Senior Quant Audit.*

| Risk | Mitigation Strategy |
| :--- | :--- |
| **Latency Trap** | **Decoupling**: Run "Cortex" (Visuals/Calculations) in a separate `multiprocessing.Process` or completely separate service from the `Execution Loop`. |
| **Genome Overfitting** | **Slow Learning**: Enforce strict `MIN_HISTORY` (>30 events) and low `LEARNING_RATE` (0.01). Use "Family-level" evolution, not asset-level. |
| **Complexity Entropy** | **Audit Trails**: Implement `neural_state.json` to log *every* decision layer's input/output for every tick. |
| **Execution Errors** | **State Machine**: Implement a strict `ExecutionStateMachine` (IDLE -> READY -> EXECUTE) to prevent loops and duplicate orders. |

---

## 📅 Roadmap

### Phase 1: The Event-Driven Core & Plumbing (Foundation)

*Goal: Move from Polling to Streaming without breaking things.*

- [ ] **Websocket Engine**: Implement `broker_angleone_ws.py` for live tick streaming.
- [ ] **State Machine**: Create `execution_state_machine.py` to manage trade lifecycle states.
- [ ] **Telemetry Standard**: Define the `neural_state.json` structure for strict logging.

### Phase 2: The Physics & Signal Engine (The Eyes)

*Goal: Ensure the "Econophysics" logic is correctly calculating on live ticks.*

- [ ] **Live Physics**: Port `get_energy`, `check_structure` (Vacuum), `get_gaussian` into the Event Loop.
- [ ] **Signal Buffer**: Create a rolling window buffer to calculate these metrics in real-time (vs loading entire CSVs).

### Phase 3: The Governance Layers (The Brain)

*Goal: Implement the hierarchy of decision making.*

- [ ] **Regime Drift Detector**: Implement Entropy/Distribution shift detection.
- [ ] **Strategy Genome**: Create the parameter evolution module (updates slowly).
- [ ] **Capital Allocation Brain**: Implement the Logic that scales `LEVERAGE` based on "Stability".
- [ ] **Meta Risk**: Implement the "Veto" layer (Rolling Sharpe/Drawdown checks).

### Phase 4: The Cortex (The Consciousness)

*Goal: Visualization and "Self-Awareness" without adding latency.*

- [ ] **Unified Cortex**: Create a separate dashboard (Streamlit) that reads `neural_state.json` to visualize the "Brain".
- [ ] **Neural Map**: Visualize the influence weights of different layers.

### Phase 5: Smart Execution (The Hands)

*Goal: Execute the "Brain's" will with finesse.*

- [ ] **Smart Execution**: Integrate the Throttler, VWAP Slicer, and Liquidity Adjuster (already prototyped).
- [ ] **Async Router**: Connect the State Machine to the Async Broker.

---

## 📂 Target Architecture

```
Chimera_Pro_Live/
├── data/                       # Local Storage
│   └── neural_state.json       # The "Black Box" Recorder
├── chimera_core/
│   ├── event_loop.py           # Main Websocket Handler
│   ├── state_machine.py        # Execution Governance
│   └── physics_engine.py       # Signal Calculations
├── chimera_governance/
│   ├── genome.py               # Parameter Evolution
│   ├── drift_detector.py       # Regime Change
│   ├── capital_brain.py        # Leverage Sizing
│   └── meta_risk.py            # Circuit Breakers
├── chimera_execution/          # (Existing)
│   ├── smart_execution.py
│   └── broker_angleone_ws.py
└── tools/
    └── cortex_dashboard.py     # The Visualization Layer
```

## 🚀 Immediate Next Step

Begin **Phase 1**: Build the `broker_angleone_ws.py` and the `execution_state_machine.py`.
