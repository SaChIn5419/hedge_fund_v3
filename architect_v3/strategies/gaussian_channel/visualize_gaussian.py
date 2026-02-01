import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

print("ARCHITECT: Reconstructing Equity Physics from Trade Log...")

# 1. LOAD TRADE LOG
df = pl.read_csv("trade_log.csv")

# Ensure date is properly parsed
df = df.with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S.%f"))

# 2. AGGREGATE TO PORTFOLIO LEVEL (DAILY)
portfolio = (
    df.group_by("date")
    .agg([
        pl.col("net_pnl").sum().alias("daily_pnl"),
        pl.col("energy").mean().alias("avg_energy") # Market Kinetic Energy
    ])
    .sort("date")
)

# 3. CALCULATE EQUITY CURVE & GAUSSIAN CHANNEL
INITIAL_CAPITAL = 1_000_000 # 10 Lakhs
ROLLING_WINDOW = 60 # 3 Months

portfolio = portfolio.with_columns([
    (pl.lit(INITIAL_CAPITAL) + pl.col("daily_pnl").cum_sum()).alias("equity")
])

portfolio = portfolio.with_columns([
    pl.col("equity").rolling_mean(window_size=ROLLING_WINDOW).alias("sma"),
    pl.col("equity").rolling_std(window_size=ROLLING_WINDOW).alias("std")
]).with_columns([
    (pl.col("sma") + (2 * pl.col("std"))).alias("upper_band"),
    (pl.col("sma") - (2 * pl.col("std"))).alias("lower_band")
]).drop_nulls()

# 4. PLOTTING PHYSICS ENGINE (Plotly)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7, 0.3])

dates = portfolio["date"].to_list()
equity = portfolio["equity"].to_list()

# --- TOP: Equity Curve + Gaussian Channel ---
# The Channel
fig.add_trace(go.Scatter(x=dates, y=portfolio["upper_band"].to_list(), line=dict(color='rgba(255,255,255,0)'), showlegend=False, name='Upper'), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=portfolio["lower_band"].to_list(), line=dict(color='rgba(255,255,255,0)'), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)', showlegend=False, name='Gaussian Band'), row=1, col=1)

# The SMA (Mean)
fig.add_trace(go.Scatter(x=dates, y=portfolio["sma"].to_list(), line=dict(color='#e67e22', width=1, dash='dot'), name='60-Day Mean'), row=1, col=1)

# The Equity Curve
fig.add_trace(go.Scatter(x=dates, y=equity, line=dict(color='#0984e3', width=2), name='Strategy Equity'), row=1, col=1)

# IDENTIFY BREAKDOWNS (Where Equity drops below the lower band)
breakdown_mask = portfolio["equity"] < portfolio["lower_band"]
breakdown_dates = portfolio.filter(breakdown_mask)["date"].to_list()
breakdown_equity = portfolio.filter(breakdown_mask)["equity"].to_list()

fig.add_trace(go.Scatter(x=breakdown_dates, y=breakdown_equity, mode='markers', 
                         marker=dict(color='red', size=6, symbol='x'), 
                         name='Circuit Breaker (Sell to Cash)'), row=1, col=1)


# --- BOTTOM: Kinetic Energy Tracker ---
fig.add_trace(go.Bar(x=dates, y=portfolio["avg_energy"].to_list(), marker_color='#2ecc71', name='Kinetic Energy'), row=2, col=1)

# Format the chart
fig.update_layout(
    title="ARCHITECT: Equity Gaussian Channel & Energy Overlay",
    template="plotly_dark",
    height=800,
    hovermode="x unified"
)
fig.update_yaxes(title_text="Total Equity (INR)", row=1, col=1)
fig.update_yaxes(title_text="Kinetic Energy", type="log", row=2, col=1) # Log scale for energy

# 5. RENDER
output_file = "energy_gaussian_overlay.html"
fig.write_html(output_file)
print(f"SUCCESS: Opening '{output_file}' in browser.")
webbrowser.open('file://' + os.path.realpath(output_file))
