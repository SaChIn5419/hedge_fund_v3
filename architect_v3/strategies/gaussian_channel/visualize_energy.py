import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

print("ARCHITECT: Reconstructing Equity Physics from Trade Log...")

# 1. LOAD TRADE LOG
df = pl.read_csv("trade_log.csv")
df = df.with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S.%f"))

# 2. AGGREGATE TO DAILY PORTFOLIO
portfolio = (
    df.group_by("date")
    .agg([
        pl.col("net_pnl").sum().alias("daily_pnl"),
        pl.col("energy").mean().alias("avg_energy") # Market Kinetic Energy
    ])
    .sort("date")
)

# 3. CALCULATE EQUITY CURVE & GARCH-STYLE GAUSSIAN CHANNEL
INITIAL_CAPITAL = 1_000_000 
# Using Span=60 for Exponential Weighting (reacts faster to crashes than Simple MA)
SPAN = 60 

portfolio = portfolio.with_columns([
    (pl.lit(INITIAL_CAPITAL) + pl.col("daily_pnl").cum_sum()).alias("equity")
])

# The GARCH/EVT philosophy: Volatility is dynamic and heavily weighted to the recent past.
portfolio = portfolio.with_columns([
    pl.col("equity").ewm_mean(span=SPAN).alias("ewm_mean"),
    pl.col("equity").ewm_std(span=SPAN).alias("ewm_std")
]).with_columns([
    (pl.col("ewm_mean") + (2.5 * pl.col("ewm_std"))).alias("upper_band"),
    (pl.col("ewm_mean") - (2.5 * pl.col("ewm_std"))).alias("lower_band")
]).drop_nulls()

# 4. PLOTTING PHYSICS

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7, 0.3])

dates = portfolio["date"].to_list()

# --- TOP: Equity + Gaussian Channel ---
fig.add_trace(go.Scatter(x=dates, y=portfolio["upper_band"].to_list(), line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=portfolio["lower_band"].to_list(), line=dict(color='rgba(255,255,255,0)'), fill='tonexty', fillcolor='rgba(231, 76, 60, 0.1)'), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=portfolio["ewm_mean"].to_list(), line=dict(color='#e67e22', width=1, dash='dot'), name='EWMA Mean'), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=portfolio["equity"].to_list(), line=dict(color='#0984e3', width=2), name='Strategy Equity'), row=1, col=1)

# CIRCUIT BREAKER TRIGGERS
breakdown = portfolio.filter(pl.col("equity") < pl.col("lower_band"))
fig.add_trace(go.Scatter(x=breakdown["date"].to_list(), y=breakdown["equity"].to_list(), 
                         mode='markers', marker=dict(color='red', size=8, symbol='x'), 
                         name='Circuit Breaker Triggered'), row=1, col=1)

# --- BOTTOM: Kinetic Energy Tracker ---
fig.add_trace(go.Bar(x=dates, y=portfolio["avg_energy"].to_list(), marker_color='#2ecc71', name='Kinetic Energy'), row=2, col=1)

fig.update_layout(title="ARCHITECT: Equity Gaussian Channel & Energy Dynamics", template="plotly_dark", height=800)
fig.update_yaxes(title_text="Total Equity (INR)", row=1, col=1)
fig.update_yaxes(title_text="Kinetic Energy (Log)", type="log", row=2, col=1)

output_file = "energy_gaussian_overlay.html"
fig.write_html(output_file)
print(f"SUCCESS: Opening '{output_file}'.")
webbrowser.open('file://' + os.path.realpath(output_file))
