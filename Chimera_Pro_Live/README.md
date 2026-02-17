# CHIMERA PRO LIVE (Institutional Grade)

This directory contains the fully isolated, production-ready environment for the **Chimera Trading Protocol**.

## 📂 Structure

- **`chimera_protocol/`**: The Research & Risk Layer.
  - `chimera_engine.py`: The original research engine.
  - `chimera_live.py`: The live risk protocol (Traffic Light System).
  
- **`chimera_execution/`**: The Institutional Execution Layer.
  - `signal_generator.py`: Generates live signals (Physics + Regime + Selection).
  - `smart_execution.py`: Throttling, Slicing (VWAP), Retry Logic.
  - `main_execution.py`: The Master Controller connecting Signal -> Risk -> Broker.
  - `config.py`: System configuration (API Keys, Tickers, Risk parameters).

- **`data/`**: Local data storage (Risk logs, trade logs).

## 📦 Installation

This environment requires specific libraries. Install them using:

```bash
pip install -r requirements.txt
```

Note: `smartapi-python` is required for Angel One integration. If you don't have it, clone it from the official repo or install via pip.

## 🚀 How to Run

### 1. Daily Monitoring (The Dashboard)

To view the system status, risk metrics, and historical replay:

```bash
python run_live_monitor.py
```

### 2. Live Execution (The Engine)

To generate signals and execute trades (requires API Keys in `chimera_execution/config.py`):

```bash
python chimera_execution/main_execution.py
```

## ⚠️ Pre-Flight Check

1. Ensure **Angel One API Keys** are set in `chimera_execution/config.py`.
2. Ensure you have a stable internet connection for Data Fetching & Broker API.
3. Check `run_live_monitor.py` for system health before executing.
