import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- CONFIGURATION ---
CSV_PATH = "chimera_blackbox_final.csv" # Ensure this matches your Phase 8.5 output
PATHS = 25000
YEARS = 5
BLOCK_SIZE = 20 # Resample 20-day chunks to preserve "Trend Memory"
INITIAL_CAPITAL = 1000000

class ChimeraMultiverse:
    def __init__(self, csv_path):
        print("--- INITIALIZING MULTIVERSE ENGINE ---")
        try:
            # Check if file exists to avoid immediate crash
            if not os.path.exists(csv_path):
                 raise FileNotFoundError(f"{csv_path} not found")
                 
            self.df = pd.read_csv(csv_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
            print(f"Loaded Trade Log: {len(self.df)} rows")
        except FileNotFoundError:
            print("[ERROR] CSV not found. Run strategy_chimera_final.py first.")
            self.daily_returns = np.array([])
            return

        self.prepare_returns()

    def prepare_returns(self):
        """
        Converts trade log into a daily % return series.
        """
        print("Constructing Return Distribution Matrix...")
        
        # Check if we have the necessary columns for Real Data Simulation
        if 'weight' in self.df.columns and 'fwd_return' in self.df.columns:
            print(">>> SOURCE: REAL TRADING DATA DETECTED <<<")
            
            # Calculate daily weighted return for the portfolio
            # Sum of (Weight * Return) for each day across all active tickers
            daily_returns_series = self.df.groupby('date').apply(lambda x: (x['weight'] * x['fwd_return']).sum())
            
            self.daily_returns = daily_returns_series.values
            
            # Basic sanity check
            print(f"Data Points: {len(self.daily_returns)}")
            print(f"Mean Daily Return: {np.mean(self.daily_returns):.4%}")
            print(f"Daily Volatility:  {np.std(self.daily_returns):.4%}")
            
        else:
            print(">>> SOURCE: MOCK SIMULATION (Real columns missing) <<<")
            # Simulating the statistical properties of "Chimera" (Fallback)
            # 1. Bull Days (Small wins, occasional big wins)
            bull_returns = np.random.normal(0.0015, 0.01, 1000) 
            # 2. Bear Days (Small losses due to defensive mode, NO big crashes)
            bear_returns = np.random.normal(-0.0005, 0.005, 500)
            # 3. Turbo Days (Fat tail upside)
            turbo_returns = np.random.normal(0.02, 0.015, 100)
            
            self.daily_returns = np.concatenate([bull_returns, bear_returns, turbo_returns])
            np.random.shuffle(self.daily_returns)
        
        print(f"Distribution Loaded: {len(self.daily_returns)} data points.")

    def run_simulation(self):
        if len(self.daily_returns) == 0: return

        print(f"--- SIMULATING {PATHS} FUTURES ({YEARS} YEARS) ---")
        trading_days = 252 * YEARS
        
        # VECTORIZED BLOCK BOOTSTRAP (The Heavy Lifting)
        # We need (PATHS, TRADING_DAYS)
        
        # 1. How many blocks do we need?
        n_blocks = int(np.ceil(trading_days / BLOCK_SIZE))
        
        # 2. Generate random start indices for blocks
        # We pick random start points from history
        max_start_idx = len(self.daily_returns) - BLOCK_SIZE
        # Ensure we have enough data for at least one block
        if max_start_idx < 0:
             print(f"[ERROR] Not enough history ({len(self.daily_returns)}) for Block Size {BLOCK_SIZE}.")
             return

        random_starts = np.random.randint(0, max_start_idx, size=(PATHS, n_blocks))
        
        # 3. Construct the matrix
        sim_returns = np.zeros((PATHS, n_blocks * BLOCK_SIZE))
        
        for i in range(n_blocks):
            # For each block step, pull the slice from history for all paths
            starts = random_starts[:, i] 
            
            for p in range(PATHS):
                idx = starts[p]
                sim_returns[p, i*BLOCK_SIZE : (i+1)*BLOCK_SIZE] = self.daily_returns[idx : idx+BLOCK_SIZE]
                
        # Trim to exact days
        sim_returns = sim_returns[:, :trading_days]
        
        # 4. Calculate Equity Curves
        # Cumulative Product: Initial * (1 + r).cumprod()
        self.equity_curves = INITIAL_CAPITAL * np.cumprod(1 + sim_returns, axis=1)
        
        # 5. Calculate Metrics
        final_equity = self.equity_curves[:, -1]
        cagrs = (final_equity / INITIAL_CAPITAL) ** (1/YEARS) - 1
        
        # Max Drawdown per path
        peaks = np.maximum.accumulate(self.equity_curves, axis=1)
        drawdowns = (self.equity_curves - peaks) / peaks
        max_dds = np.min(drawdowns, axis=1)
        
        self.metrics = {
            'CAGR': cagrs,
            'MaxDD': max_dds,
            'Final_Eq': final_equity
        }
        
    def visualize(self):
        if not hasattr(self, 'equity_curves'): return

        # Calculate Percentiles
        percentiles = [5, 25, 50, 75, 95]
        perc_curves = np.percentile(self.equity_curves, percentiles, axis=0)
        
        days = np.arange(self.equity_curves.shape[1])
        
        fig = make_subplots(rows=2, cols=2, 
                            specs=[[{"colspan": 2}, None], [{"type": "histogram"}, {"type": "histogram"}]],
                            subplot_titles=("The Cone of Uncertainty (25k Paths)", "CAGR Distribution", "Max Drawdown Distribution"))
        
        # 1. THE CONE (Fan Chart)
        # 95th Percentile (Best Case)
        fig.add_trace(go.Scatter(x=days, y=perc_curves[4], mode='lines', line=dict(width=0), showlegend=False, name='95th'), row=1, col=1)
        
        # Fill to 5th Percentile (Worst Case)
        fig.add_trace(go.Scatter(x=days, y=perc_curves[0], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 204, 0.1)', showlegend=True, name='90% Confidence Interval'), row=1, col=1)
        
        # Median (Expected)
        fig.add_trace(go.Scatter(x=days, y=perc_curves[2], mode='lines', line=dict(color='#00ffcc', width=2), name='Median Outcome'), row=1, col=1)

        # 2. PROBABILITY DENSITY (CAGR)
        fig.add_trace(go.Histogram(x=self.metrics['CAGR'], nbinsx=50, marker_color='#0088ff', name='CAGR'), row=2, col=1)
        fig.add_vline(x=0, line_dash="dot", line_color="white", row=2, col=1)
        
        # 3. PROBABILITY DENSITY (Drawdown)
        fig.add_trace(go.Histogram(x=self.metrics['MaxDD'], nbinsx=50, marker_color='#ff5555', name='Max DD'), row=2, col=2)
        
        # METRICS CALCULATIONS
        ruin_prob = np.mean(self.metrics['Final_Eq'] < INITIAL_CAPITAL) * 100
        median_cagr = np.median(self.metrics['CAGR']) * 100
        worst_case_dd = np.percentile(self.metrics['MaxDD'], 5) * 100
        
        title_text = f"CHIMERA MULTIVERSE | Ruin Probability: {ruin_prob:.2f}% | Median CAGR: {median_cagr:.1f}% | 95% Var DD: {worst_case_dd:.1f}%"
        
        fig.update_layout(template="plotly_dark", title=title_text, height=800)
        fig.write_html("chimera_monte_carlo.html")
        print(f"REPORT GENERATED: {title_text}")

if __name__ == "__main__":
    mc = ChimeraMultiverse(CSV_PATH)
    mc.run_simulation()
    mc.visualize()
