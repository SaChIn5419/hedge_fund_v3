# The Chimera Protocol (Architect V3)

> **"If the order of market cycles was different, would I still survive?"**

The **Chimera Protocol** is an institutional-grade, quantitative trading system designed for robust survival and asymmetric upside in Indian Equity Markets. Unlike traditional strategies that rely on Gaussian assumptions (Bell Curves), Chimera is built on **Complexity Theory** and **Regime Detection**.

It utilizes a **"Black Box"** interaction model where the Strategy Engine, Risk Manager (Watchdog), and Simulation Layer (Multiverse) operate as independent, decoupled systems.

---

## 🏗️ System Architecture

The system follows a strict **Separation of Concerns** principle:

```mermaid
graph TD
    subgraph "The Real World (Data)"
        NSE[NSE Market Data] -->|OHLCV| Feed[Data Ingestion]
        Feed -->|Write| DB[(Data Lake / CSV)]
    end

    subgraph "Core Engines"
        DB -->|Read| Strat[Chimera Strategy Engine]
        Strat -->|Decision Loop| Logic{Regime Filter}
        Logic -->|Bull/Turbo| Buy[Select Top 5 Momentum]
        Logic -->|Bear/Crash| Cash[Liquidate to Cash/Gold]
        
        Buy -->|Generate| Log[Trade Log (Black Box)]
        Cash -->|Generate| Log
    end

    subgraph "Risk Daemon"
        NSE -->|Live Stream| Watch[Watchdog Daemon]
        Watch -->|VIX Spike / Crash| Kill[Circuit Breaker]
        Kill --INTERRUPT--> Strat
    end

    subgraph "The Multiverse (Simulation)"
        Log -->|Read Actual Returns| Sim[Reality Monte Carlo]
        Sim -->|Block Bootstrap| Paths[25,000 Parallel Futures]
        Paths -->|Analyze| Report[Forensic Risk Report]
    end
```

---

## 📂 Project Structure

The filesystem is organized to separate Logic, Data, and Analysis:

```text
architect_v3/
├── main_terminal.py            # Entry point for the trading desk
├── config.yaml                 # Global parameters (Capital, Tickers, Regimes)
│
├── strategies/
│   └── chimera_strategy.py     # THE CORE: Unlevered Momentum Strategy (Generates Black Box)
│
├── engine/
│   ├── watchdog.py             # THE DAEMON: Live Risk Monitor (Circuit Breaker)
│   ├── core_physics.py         # Physics calculations (Energy, Entropy, Structure)
│   └── execution.py            # Broker Routing Layer
│
├── tools/
│   └── chimera_monte_carlo_real.py # THE MULTIVERSE: Forensic Simulation Engine
│
├── data/
│   └── chimera_blackbox_final.csv  # The "Black Box" output (Trade Log with Forward Returns)
│
├── reports/
│   └── chimera_monte_carlo_real.html # The Final Risk Report (Cone of Uncertainty)
│
└── brokers/
    └── dhan_router.py          # API Gateway for Dhan
```

---

## 🧩 Component Deep Dive

### 1. The Chimera Strategy (`strategies/chimera_strategy.py`)

This is the heart of the system. It runs offline (or on schedule) to analyze history and generate signals.

- **Physics:** Uses `Energy` (Linear Regression Slope) and `Structure` (Volume Profile/Standard Deviation) to rank stocks.
- **Regime Detection:** Triangulates `Nifty 500` (Trend), `Smallcap 100` (Risk Appetite), and `India VIX` (Fear) to decide between **100% Equity** and **100% Cash**.
- **Forensic Logging:** Crucially, it calculates the `fwd_return` (Future Return) for every trade at the moment of generation. This allows the simulation layer to be "honest" and not guess outcomes.

### 2. The Watchdog Daemon (`engine/watchdog.py`)

A standalone, asynchronous process that guards the portfolio.

- **Duty:** Runs every minute/hour to check "Vital Signs" (VIX > 24, Nifty Drop > 3%).
- **Authority:** Has the power to **Liquidate All Positions** immediately, overriding the Strategy Engine if a crash is detected.
- **Hysteresis:** Once triggered, it keeps the system in `DEFENSIVE` mode until volatility cools down (VIX < 18).

### 3. The Multiverse Engine (`tools/chimera_monte_carlo_real.py`)

A customized **Block Bootstrap** simulation.

- **Problem:** Standard Monte Carlo assumes normal distribution (Gaussian), which underestimates "Fat Tails" (Crashes and Super-Trends).
- **Solution:** It takes the *actual* historical weekly returns from the `chimera_blackbox_final.csv`, cuts them into 8-week chunks (to preserve memory/trend), and reshuffles them 25,000 times.
- **Output:** Generates a Probability Cone showing the "Worst Case" (95% VaR) and "Ruin Probability".

---

## 🚀 How to Deploy

### Step 1: Generate the Black Box

Run the strategy to analyze history and build the trade log.

```bash
python strategies/chimera_strategy.py
```

*Output: `data/chimera_blackbox_final.csv`*

### Step 2: Run the Simulation

Stress-test the strategy against 25,000 alternative histories.

```bash
python tools/chimera_monte_carlo_real.py
```

*Output: `reports/chimera_monte_carlo_real.html`*

### Step 3: Activate the Desk

(Optional) Run the main terminal to manage live trading.

```bash
python main_terminal.py
```

### Step 4: The Sentinel

Ensure the Watchdog is running on a server/cron job during market hours.

```bash
python engine/watchdog.py
```

---

## 📊 Performance Metrics (Forensic Audit 2019-2026)

- **Average Weekly Return:** ~0.89%
- **Worst Historical Week:** -8.66%
- **Median Simulated CAGR:** >60%
- **Ruin Probability (5 Years):** 0.00%

> *"Complexity is the only way to model a complex system. Gaussian models are a map of a world that does not exist."*
