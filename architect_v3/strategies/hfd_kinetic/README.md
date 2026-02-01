# HFD Kinetic Strategy (Architect V4)

**High-Frequency Data (HFD) Regime Switching Strategy**

This folder contains the self-contained Logic, Backtesting Engine, and Verification Suite for the "Golden Configuration" strategy.

## File Structure

### 1. The Strategy

- **`main.py`**: The Production Strategy script. Loads data, runs the logic, and outputs the "Investment Memorandum".
  - *Usage*: `python strategies/hfd_kinetic/main.py`

### 2. Backtesting (The "Glass Box")

- **`backtesting/glass_engine.py`**: The Custom Numba Event-Driven Engine.
- **`backtesting/viz_engine.py`**: The Institutional Reporting Module (Sharpe, Sortino, SQN).

### 3. Verification & Tests

- **`tests/stress_test_oos.py`**: Out-of-Sample "Chop" validation (0 Trades expected).
- **`tests/stress_test_regimes.py`**: Multi-Regime Switch validation (Trend vs Chop).
- **`tests/monte_carlo.py`**: Massive Risk Analysis (1k-25k runs).

### 4. Optimization

- **`optimization/hyper_opt.py`**: The Grid Search script that found the parameters.

### 5. Legacy

- **`legacy/regime_swing_v3.py`**: The prototype that led to this version.

## Strategy Logic (Golden Config)

- **Timeframe**: 1-Minute Aggregation (from Tick/Second Data).
- **Filters**:
  - `FDI < 1.5`: Market must be organized (Trend/Structure).
  - `Alpha < 1.6`: Market must have Fat Tails (Levy Flight).
- **Execution**: Long Only.
- **Risk**:
  - Exit when `FDI > 1.6` (Regime decays to Random Noise).

## Performance (Audit)

- **Sharpe Ratio**: > 4.0
- **Win Rate**: > 70%
- **Profit Factor**: > 3.0
- **Risk of Ruin**: 0.00% (Monte Carlo verified)
