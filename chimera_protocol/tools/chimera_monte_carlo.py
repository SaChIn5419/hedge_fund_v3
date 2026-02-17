import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
CSV_PATH = "data/chimera_blackbox_final.csv" 
PATHS = 25000         # Number of alternative futures
YEARS = 5             # Simulation duration
BLOCK_SIZE = 8        # 8 Weeks (~2 Months) to preserve trend memory
INITIAL_CAPITAL = 1000000

class ChimeraMultiverseReal:
    def __init__(self, csv_path):
        print("--- INITIALIZING REALITY-BASED MULTIVERSE ---")
        try:
            self.df = pd.read_csv(csv_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
        except FileNotFoundError:
            print(f"[ERROR] Could not find {csv_path}")
            return

        self.weekly_returns = []
        self.prepare_returns()

    def prepare_returns(self):
        """
        Reconstructs the Portfolio-Level Weekly Return series from the trade log.
        """
        print("Reconstructing Historical Portfolio Returns...")
        
        # 1. Clean Data
        # Ensure we have fwd_return (fill NaNs with 0 for Cash/Safety)
        if 'fwd_return' not in self.df.columns:
            print("[CRITICAL] 'fwd_return' column missing in CSV.")
            return
            
        self.df['fwd_return'] = self.df['fwd_return'].fillna(0.0)
        
        # 2. Calculate Weighted Return per Date
        # Portfolio_Ret = Sum(Weight * Asset_Ret)
        # Note: If strategy is in CASH (weight=1.0), returns are 0.0 (or risk-free if modeled).
        # We assume the log contains all positions. If cash is not explicit, sum of weights might be < 1.0.
        # This implies the remainder is Cash (return 0).
        
        # Group by date
        # WARNING: This assumes 'fwd_return' is the return of the ASSET for that period
        portfolio_series = self.df.groupby('date').apply(
            lambda x: np.sum(x['weight'] * x['fwd_return'])
        )
        
        self.weekly_returns = portfolio_series.values
        
        # Basic Stats check
        avg_ret = np.mean(self.weekly_returns)
        std_ret = np.std(self.weekly_returns)
        print(f"History Loaded: {len(self.weekly_returns)} weeks.")
        print(f"Avg Weekly Ret: {avg_ret:.2%} | Std Dev: {std_ret:.2%}")
        
        # Sanity Check for 'Crash' weeks
        worst_week = np.min(self.weekly_returns)
        print(f"Worst Historical Week: {worst_week:.2%}")

    def run_simulation(self):
        if len(self.weekly_returns) < BLOCK_SIZE:
            print("[ERROR] Not enough history for Block Bootstrap.")
            return

        print(f"--- SIMULATING {PATHS} FUTURES ({YEARS} YEARS) ---")
        
        # SIMULATION PHYSICS: WEEKLY
        # 5 Years * 52 Weeks = 260 Steps
        steps = 52 * YEARS 
        
        # VECTORIZED BLOCK BOOTSTRAP
        n_blocks = int(np.ceil(steps / BLOCK_SIZE))
        
        # Randomly select start indices for blocks from the historical array
        max_start = len(self.weekly_returns) - BLOCK_SIZE
        if max_start <= 0:
            print("History too short for requested Block Size.")
            return
            
        # Matrix of random start indices: (PATHS, n_blocks)
        starts = np.random.randint(0, max_start, size=(PATHS, n_blocks))
        
        # Construct the returns matrix
        # Shape: (PATHS, n_blocks * BLOCK_SIZE)
        # We will slice to 'steps' later
        sim_returns = np.zeros((PATHS, n_blocks * BLOCK_SIZE))
        
        for i in range(n_blocks):
            # For each block column, copy the slices from history
            block_starts = starts[:, i]
            
            # This loop is fast enough for 25k paths
            for p in range(PATHS):
                s = block_starts[p]
                sim_returns[p, i*BLOCK_SIZE : (i+1)*BLOCK_SIZE] = self.weekly_returns[s : s+BLOCK_SIZE]
                
        # Trim excess weeks
        sim_returns = sim_returns[:, :steps]
        
        # CALCULATE EQUITY CURVES
        # Cumulative Product: Start * (1 + r1) * (1 + r2)...
        self.equity_curves = INITIAL_CAPITAL * np.cumprod(1 + sim_returns, axis=1)
        
        # METRICS
        final_equity = self.equity_curves[:, -1]
        
        # CAGR = (End/Start)^(1/Years) - 1
        self.cagrs = (final_equity / INITIAL_CAPITAL) ** (1/YEARS) - 1
        
        # Max Drawdown
        peaks = np.maximum.accumulate(self.equity_curves, axis=1)
        drawdowns = (self.equity_curves - peaks) / peaks
        self.max_dds = np.min(drawdowns, axis=1)
        
        self.final_equity = final_equity

    def visualize(self):
        if not hasattr(self, 'cagrs'): return

        # Percentiles for the Fan Chart
        percentiles = [5, 25, 50, 75, 95]
        perc_curves = np.percentile(self.equity_curves, percentiles, axis=0)
        weeks = np.arange(self.equity_curves.shape[1])
        
        fig = make_subplots(
            rows=2, cols=2, 
            specs=[[{"colspan": 2}, None], [{"type": "histogram"}, {"type": "histogram"}]],
            subplot_titles=("The Multiverse: 25,000 Possible Futures", "CAGR Probability", "Drawdown Risk")
        )
        
        # 1. FAN CHART
        # 95th (Best)
        fig.add_trace(go.Scatter(x=weeks, y=perc_curves[4], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
        # 95th-5th Fill (Confidence Interval)
        fig.add_trace(go.Scatter(x=weeks, y=perc_curves[0], mode='lines', line=dict(width=0), 
                                 fill='tonexty', fillcolor='rgba(0, 255, 204, 0.1)', name='90% Confidence Interval'), row=1, col=1)
        # Median
        fig.add_trace(go.Scatter(x=weeks, y=perc_curves[2], mode='lines', 
                                 line=dict(color='#00ffcc', width=2), name='Median Outcome'), row=1, col=1)
        
        # 2. CAGR HISTOGRAM
        fig.add_trace(go.Histogram(x=self.cagrs * 100, nbinsx=100, marker_color='#0088ff', name='CAGR %'), row=2, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="white", row=2, col=1)

        # 3. DRAWDOWN HISTOGRAM
        fig.add_trace(go.Histogram(x=self.max_dds * 100, nbinsx=100, marker_color='#ff5555', name='Max DD %'), row=2, col=2)
        
        # STATISTICS
        ruin_prob = np.mean(self.final_equity < INITIAL_CAPITAL) * 100
        median_cagr = np.median(self.cagrs) * 100
        var_95_dd = np.percentile(self.max_dds, 5) * 100 # The 5th percentile worst case
        
        title_text = f"REALITY BOOTSTRAP | Median CAGR: {median_cagr:.1f}% | 95% Var DD: {var_95_dd:.1f}% | Ruin Prob: {ruin_prob:.2f}%"
        
        fig.update_layout(template="plotly_dark", title=title_text, height=900)
        
        filename = "chimera_monte_carlo_real.html"
        fig.write_html(filename)
        print(f"REPORT GENERATED: {filename}")

if __name__ == "__main__":
    mc = ChimeraMultiverseReal(CSV_PATH)
    mc.run_simulation()
    mc.visualize()
