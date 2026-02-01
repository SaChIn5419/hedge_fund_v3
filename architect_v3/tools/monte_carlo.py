import polars as pl
import numpy as np
import plotly.graph_objects as go
import webbrowser
import os
import sys

# Add parent directory to path if needed, though this script seems standalone mostly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("ARCHITECT: Initiating 25,000-Path Monte Carlo Matrix...")

# 1. LOAD ACTUAL STRATEGY RETURNS (Bootstrapping)
# We sample from your actual strategy's historical returns, including the fat tails.
# Assuming 'trade_log.csv' is generated from the 27% CAGR run
try:
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'trade_log.csv')
    df = pl.read_csv(csv_path)
    df = df.with_columns(
        pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f")
    )
except Exception as e:
    print(f"Error loading trade_log.csv: {e}")
    exit()

# Aggregate to Portfolio Daily Returns
daily_portfolio = (
    df.group_by("date")
    .agg(pl.col("net_pnl").sum().alias("daily_pnl"))
    .sort("date")
)

# Convert PnL to percentage returns
# Infer Initial Capital from the data data to handle different backtest runs (IS vs OOS vs Full)
# position_value = weight * initial_capital  =>  initial_capital = position_value / weight
# We take the median to filter out potential outliers
if "position_value" in df.columns and "weight" in df.columns:
    inferred_capital = (df["position_value"] / df["weight"]).median()
    print(f"ARCHITECT: Inferred base capital from trade log: INR {inferred_capital:,.0f}")
else:
    inferred_capital = 1_000_000
    print("ARCHITECT: Could not infer capital, defaulting to INR 1,000,000")

# Note: In the backtest, we might have periods of cash. For MC, we sample the active trading days.
# Or better, we calculate return relative to capital.
daily_returns_pct = (daily_portfolio["daily_pnl"] / inferred_capital).to_numpy()

# 2. THE PHYSICS MATRIX (NUMPY BROADCASTING)
# Simulating 5 years (1250 days) across 25,000 alternate universes.
# RAM Cost: ~250 MB (Handled by Numpy in < 1 second)
DAYS = 1250
SIMULATIONS = 25000

print(f"ARCHITECT: Generating {SIMULATIONS * DAYS:,.0f} data points from {len(daily_returns_pct)} historical samples...")

# Resample historical returns with replacement
simulation_matrix = np.random.choice(daily_returns_pct, size=(SIMULATIONS, DAYS), replace=True)

# Add 1 to returns for compounding
simulation_matrix += 1 

# Calculate Cumulative Equity for all 25,000 paths instantly
# np.cumprod along axis 1 calculates the compounding effect
equity_paths = inferred_capital * np.cumprod(simulation_matrix, axis=1)

# 3. STATISTICAL EXTRACTION
# We extract the 5th (Worst), 50th (Median), and 95th (Best) percentile paths
percentile_5 = np.percentile(equity_paths, 5, axis=0)
percentile_50 = np.percentile(equity_paths, 50, axis=0)
percentile_95 = np.percentile(equity_paths, 95, axis=0)

# Final Capital Distribution (for Terminal output)
final_capitals = equity_paths[:, -1]
risk_of_ruin = np.sum(final_capitals < (inferred_capital * 0.5)) / SIMULATIONS

print("\n--- MONTE CARLO RESULTS (25,000 PATHS) ---")
print(f"Median Final Equity: INR {np.median(final_capitals):,.2f}")
print(f"95th Percentile (Luck): INR {np.percentile(final_capitals, 95):,.2f}")
print(f"5th Percentile (Bad Luck): INR {np.percentile(final_capitals, 5):,.2f}")
print(f"Risk of 50% Drawdown (Terminal): {risk_of_ruin * 100:.2f}%")
print("------------------------------------------")

# 4. PLOTTING THE PROBABILITY CONE
fig = go.Figure()

x_days = np.arange(DAYS)

# Plot a "Ghost" background of 100 random paths to show the chaos
for i in range(100):
    fig.add_trace(go.Scatter(x=x_days, y=equity_paths[i, :], mode='lines', 
                             line=dict(color='rgba(100, 100, 100, 0.05)'), showlegend=False))

# Plot the 95th Percentile (Upper Bound)
fig.add_trace(go.Scatter(x=x_days, y=percentile_95, mode='lines', name='95th Percentile (Bull)', 
                         line=dict(color='#2ecc71', width=2, dash='dash')))

# Plot the Median
fig.add_trace(go.Scatter(x=x_days, y=percentile_50, mode='lines', name='50th Percentile (Expected)', 
                         line=dict(color='#3498db', width=3)))

# Plot the 5th Percentile (Lower Bound)
fig.add_trace(go.Scatter(x=x_days, y=percentile_5, mode='lines', name='5th Percentile (Bear)', 
                         line=dict(color='#e74c3c', width=2, dash='dash')))

fig.update_layout(
    title="ARCHITECT V3: 25,000-Path Monte Carlo Simulation (5-Year Horizon)",
    template="plotly_dark",
    xaxis_title="Trading Days (5 Years)",
    yaxis_title="Portfolio Equity (INR)",
    showlegend=True
)

# 5. RENDER
# 5. RENDER
output_file = os.path.join(os.path.dirname(__file__), '..', 'reports', "monte_carlo_cone.html")
fig.write_html(output_file)
print(f"SUCCESS: Opening '{output_file}'.")
webbrowser.open('file://' + os.path.realpath(output_file))
