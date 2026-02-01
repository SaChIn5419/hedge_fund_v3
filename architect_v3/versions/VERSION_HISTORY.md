# ARCHITECT V3 Version History

# Each version is a snapshot of working code that can be restored

## v3_golden_ratio (2026-01-26 23:30) [STABLE]

- **Metrics:** +27.64% CAGR, -81.34% DD, +1116% Total Return
- **Configuration:**
  - Close-to-Close Execution (T+0, capturing overnight gaps)
  - Late-Stage Nifty 200-SMA Regime Filter (Portfolio Level)
  - 15% Rolling Trailing Stop-Loss (Chandelier Exit)
- **Status:** The highest performing model. The drawdown is high, but the alpha is maximized.

## v1_stable (2026-01-26 22:46)

- Working backtest from 2015-present
- Reality-capped with liquidity filters
- Hyper-grid tear sheet with trade visualization
- Trade log export with Gaussian channel data

### Files in v3_golden_ratio

- core_physics.py
- analytics.py  
- execution.py
- alpha_v1.py
- main_terminal.py
- config.yaml
- visualize_energy.py

### To Restore

```powershell
Copy-Item -Path "versions\v3_golden_ratio\*" -Destination "." -Force
Copy-Item -Path "versions\v3_golden_ratio\core_physics.py" -Destination "engine\" -Force
```
