import polars as pl
import numpy as np
import pandas as pd
import warnings
import sys
import os
from numba import jit

# Import Glass Engine and Kernels from the existing file
# Ensure we are in the correct directory for imports
sys.path.append(os.getcwd())
try:
    from engine.glass_engine import GlassEngine, _numba_rolling_fdi, _numba_rolling_hill, run_event_loop
except ImportError:
    from glass_engine import GlassEngine, _numba_rolling_fdi, _numba_rolling_hill, run_event_loop

# Windows Console Encoding Fix
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

warnings.filterwarnings('ignore')

@jit(nopython=True)
def generate_regime_data(n):
    prices = np.zeros(n)
    prices[0] = 1600.0
    
    cutoff = int(n * 0.5)
    
    # 1. THE CHOP (Ornstein-Uhlenbeck)
    # High Noise, Mean Reverting
    mu, theta, sigma = 1600.0, 0.1, 2.0
    dt_inv_n = 1.0 / n # Matching user logic
    
    # Pre-generate random noise for speed
    # We can't use random.standard_t inside numba easily without newer versions?
    # standard_t is available in numpy, we can pass it in.
    # But for O-U we used normal.
    
    # We'll do the loop in parts.
    # We need noise arrays passed in. function will just iterate.
    return prices # Placeholder, see actual implementation below

@jit(nopython=True)
def fast_regime_gen(n, cutoff, noise_normal, noise_fat_tail):
    prices = np.zeros(n)
    prices[0] = 1600.0
    x = 1600.0
    
    # Params
    mu = 1600.0
    theta = 0.01 # Reversion strength
    sigma = 5.0 # Volatility
    dt = 1.0 # 1 second
    sqrt_dt = 1.0
    
    # 1. THE CHOP (Iterative O-U)
    # dx = theta*(mu-x)*dt + sigma*dW
    for t in range(1, cutoff):
        dx = theta * (mu - x) * dt + sigma * noise_normal[t]
        x = x + dx
        if x < 100: x = 100 # Floor price
        prices[t] = x
        
    # 2. THE TREND (Random Walk with Drift + Fat Tails)
    trend_drift = 0.05 # 5 cents per second
    
    for t in range(cutoff, n):
        shock = noise_fat_tail[t] * 2.0 
        x = x + trend_drift + shock
        if x < 100: x = 100
        prices[t] = x
        
    return prices

def generate_positive_control_data():
    print("[ARCHITECT] Generating Multi-Regime Stress Test...")
    dates = pd.date_range(start="2023-02-01", end="2023-04-01", freq="1s")
    n = len(dates)
    cutoff = int(n * 0.5)
    
    print(f"[ARCHITECT] Synthesis: {n} ticks. Split at {cutoff} (Chop -> Trend).")
    
    print("[ARCHITECT] Generating Noise Vectors...")
    noise_normal = np.random.normal(0, 1.0, n)
    # Student-t (df=2.5)
    noise_fat_tail = np.random.standard_t(df=2.5, size=n)
    
    print("[ARCHITECT] Running Numba Synthesis...")
    prices = fast_regime_gen(n, cutoff, noise_normal, noise_fat_tail)
    
    print(f"[ARCHITECT] Price Range: Min={prices.min():.2f}, Max={prices.max():.2f}")
    
    df = pl.DataFrame({
        "time": dates.astype(np.int64) // 10**6,
        "close": prices,
        "symbol": "ETH-USD"
    })
    return df

import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ArchitectReport:
    def __init__(self, simulation_results, symbol="ETH-USD"):
        """
        :param simulation_results: Output from GlassEngine.run()
        :param symbol: Ticker symbol for display
        """
        self.symbol = symbol
        self.trades = simulation_results['trades']
        self.equity_curve = simulation_results.get('equity_curve', [])
        
    def generate_html(self, time_index, filename="Architect_Report.html"):
        print(f"[ARCHITECT] Forging HTML Dashboard: {filename}...")
        
        # Prepare Data
        equity = pd.Series(self.equity_curve, index=time_index)
        
        # Calculate Drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        
        # Metrics Calculation
        if len(equity) > 0:
            total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
            max_dd = drawdown.min()
        else:
            total_ret = 0
            max_dd = 0
        
        if len(self.trades) > 0:
            win_rate = len(self.trades[self.trades['pnl_pct'] > 0]) / len(self.trades)
            gross_win = self.trades[self.trades['pnl_pct'] > 0]['pnl_pct'].sum()
            gross_loss = abs(self.trades[self.trades['pnl_pct'] <= 0]['pnl_pct'].sum())
            profit_factor = gross_win / gross_loss if gross_loss != 0 else 0
        else:
            win_rate = 0
            profit_factor = 0

        # --- PLOTLY DASHBOARD CONSTRUCTION ---
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(f"Equity Curve ({self.symbol})", "Drawdown Profile", "Trade PnL Distribution")
        )

        # 1. EQUITY CURVE
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity, 
            mode='lines', name='Equity',
            line=dict(color='#00ff00', width=2)
        ), row=1, col=1)
        
        # Overlay Trade Markers
        if len(self.trades) > 0:
            # Approximate Y location for entry markers (on equity curve)
            # We map entry time to equity value
            # This requires reindexing trades to match equity index or using time match
            # For simplicity, we just plot markers at the time, y=Equity Value at that time?
            # Or Y=Price? Price scale is different.
            # Let's plot markers on the Equity Curve at the time of entry.
            # We need to lookup equity value at entry_time.
            # Convert entry timestamps to match equity index (Datetime)
            entry_times_dt = pd.to_datetime(self.trades['entry_time'], unit='ms')
            entry_vals = equity.asof(entry_times_dt)
            
            fig.add_trace(go.Scatter(
                x=entry_times_dt, y=entry_vals, 
                mode='markers', name='Entry',
                marker=dict(symbol='triangle-up', color='cyan', size=10)
            ), row=1, col=1)

        # 2. DRAWDOWN
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown, 
            mode='lines', name='Drawdown',
            fill='tozeroy', line=dict(color='#ff0000', width=1)
        ), row=2, col=1)

        # 3. TRADE PNL HISTOGRAM
        if len(self.trades) > 0:
            fig.add_trace(go.Histogram(
                x=self.trades['pnl_pct'] * 100,
                name='Trade PnL %',
                marker_color='#eb4034',
                opacity=0.7
            ), row=3, col=1)

        # 4. METRICS ANNOTATION
        metrics_html = f"""
        <b>ARCHITECT STRATEGY CARD</b><br>
        ---------------------------<br>
        Total Return: {total_ret*100:.2f}%<br>
        Max Drawdown: {max_dd*100:.2f}%<br>
        Win Rate:     {win_rate*100:.1f}%<br>
        Profit Factor: {profit_factor:.2f}<br>
        Trades:       {len(self.trades)}
        """
        
        fig.add_annotation(
            text=metrics_html,
            align='left',
            showarrow=False,
            xref='paper', yref='paper',
            x=0.02, y=0.98,
            bgcolor="black",
            bordercolor="cyan",
            borderwidth=1,
            font=dict(color="cyan", family="Courier New", size=12)
        )

        fig.update_layout(
            template="plotly_dark",
            height=1200,
            title_text=f"ARCHITECT V4: REGIME FILTER AUDIT - {self.symbol}",
            hovermode="x unified",
            showlegend=False
        )
        
        fig.write_html(filename)
        print(f"[SUCCESS] Dashboard generated: {filename}")

def run_positive_control():
    # ... (Existing Logic)
    # 1. Generate Data
    df_oos = generate_positive_control_data()
    
    # 2. Compute Physics (Numba)
    print("[ARCHITECT] Computing Physics...")
    df_1m = df_oos.with_columns([
        (pl.col("time") * 1000).cast(pl.Datetime("us")).alias("datetime")
    ]).group_by_dynamic("datetime", every="1m").agg([
        pl.col("close").last(),
        pl.col("time").last()
    ]).sort("datetime")
    
    close_vals = df_1m["close"].to_numpy()
    ret_vals = np.concatenate((np.array([0.0]), np.diff(close_vals))) / (close_vals + 1e-9)
    
    print("[ARCHITECT] Calculating Indicators (FDI, Alpha)...")
    fdi = _numba_rolling_fdi(close_vals, 30)
    alpha = _numba_rolling_hill(ret_vals, 100, 0.05)
    
    # 3. Run Glass Engine
    df_sim = pl.DataFrame({
        "time": df_1m["time"], 
        "close": df_1m["close"], 
        "fdi": fdi, 
        "alpha_tail": alpha
    })
    
    engine = GlassEngine(fee=0.0004, slippage=0.0001, initial_capital=100000.0)
    print("[ARCHITECT] Executing Glass Engine...")
    results = engine.run(df_sim)
    
    # 4. Analysis
    engine.tear_sheet()
    
    # 5. DASHBOARD GENERATION
    # Create time index from df_1m
    time_index = pd.to_datetime(df_1m["time"].to_numpy(), unit='ms')
    report = ArchitectReport(engine.results, "ETH-USD (Synthetic Regime Test)")
    report.generate_html(time_index, "Architect_Regime_Dashboard.html")

    return results["trades"]

if __name__ == "__main__":
    run_positive_control()
