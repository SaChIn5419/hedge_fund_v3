import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

# --- CONFIGURATION ---
LOG_FILE = 'trade_log.csv'
BENCHMARK_TICKER = '^CRSLDX'  # Adjusted to match valid usage
RISK_FREE_RATE = 0.05         # 5% for Sharpe Calc

# SCHEMA MAPPING (Correcting for headerless CSV)
COLUMNS = [
    "date", "ticker", "close", "sma_20", "upper_band", "lower_band", 
    "momentum", "energy", "volatility", "rupee_volume", 
    "alpha_weight", "market_regime", "max_weight", "weight", 
    "fwd_return", "turnover", "position_value", 
    "gross_pnl", "friction_pnl", "net_pnl"
]

class ArchitectTearsheetV11:
    def __init__(self, log_path):
        print("--- INITIATING FORENSIC AUDIT (V11) ---")
        try:
            # FIX: CSV now has headers (verified via Get-Content)
            self.df = pd.read_csv(log_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
        except Exception as e:
            print(f"[CRITICAL] Failed to load trade_log.csv: {e}")
            self.df = None
            return

    def ingest_benchmark(self):
        print(f"Fetching Benchmark ({BENCHMARK_TICKER})...")
        if self.df is None or self.df.empty: return None
        
        start = self.df['date'].min()
        end = self.df['date'].max()
        
        try:
            bench = yf.download(BENCHMARK_TICKER, start=start, end=end, progress=False)
            # Fix Multi-Index
            if isinstance(bench.columns, pd.MultiIndex):
                bench = bench.xs('Close', axis=1, level=1) if 'Close' in bench.columns.get_level_values(1) else bench.iloc[:, 0]
            elif 'Close' in bench.columns:
                bench = bench['Close']
            else:
                 bench = bench.iloc[:, 0] # Fallback
                
            bench = bench.rename('Benchmark')
            bench.index = bench.index.tz_localize(None)
            return bench
        except Exception as e:
            print(f"Benchmark Fetch Failed: {e}")
            return None

    def calculate_physics(self):
        if self.df is None: return None, 0

        # 1. Reconstruct Daily Equity Curve
        # Aggregating trade logs to portfolio level
        daily = self.df.groupby('date').agg({
            'net_pnl': 'sum',
            'friction_pnl': 'sum',
            'position_value': 'sum', # Gross Exposure
            'gross_pnl': 'sum'
        }).sort_index()
        
        # Assume starting capital from first trade context or config
        initial_capital = 1000000 
        
        daily['Equity'] = initial_capital + daily['net_pnl'].cumsum()
        daily['Returns'] = daily['Equity'].pct_change().fillna(0)
        daily['Drawdown'] = (daily['Equity'] - daily['Equity'].cummax()) / daily['Equity'].cummax()
        
        # 2. Leverage Physics (The "Dimmer Switch" Check)
        daily['Leverage'] = daily['position_value'] / daily['Equity']
        
        # 3. Efficiency Physics (Thermodynamics)
        # Efficiency = Net PnL / (Gross PnL + Abs(Friction))
        # Or simpler: Friction / Gross PnL (Burn Rate)
        total_gross = daily['gross_pnl'].sum()
        total_friction = daily['friction_pnl'].sum()
        burn_rate = (total_friction / total_gross) * 100 if total_gross != 0 else 0
        
        # 4. Benchmark Comparison
        bench = self.ingest_benchmark()
        if bench is not None:
            daily = daily.join(bench, how='inner').ffill()
            daily['Bench_Returns'] = daily['Benchmark'].pct_change().fillna(0)
            daily['Bench_Equity'] = (1 + daily['Bench_Returns']).cumprod() * initial_capital
        else:
             daily['Bench_Equity'] = initial_capital
             daily['Bench_Returns'] = 0.0

        # 5. Rolling Metrics (6-Month Window)
        window = 126
        if 'Bench_Returns' in daily.columns:
            rolling_cov = daily['Returns'].rolling(window).cov(daily['Bench_Returns'])
            rolling_var = daily['Bench_Returns'].rolling(window).var()
            daily['Beta'] = rolling_cov / rolling_var
        else:
            daily['Beta'] = 0.0
        
        daily['Rolling_Sharpe'] = (daily['Returns'].rolling(window).mean() * 252) / \
                                  (daily['Returns'].rolling(window).std() * np.sqrt(252))
                                  
        return daily, burn_rate

    def generate_dashboard(self):
        daily_stats, burn_rate = self.calculate_physics()
        if daily_stats is None: return

        # METRICS
        cagr = ((daily_stats['Equity'].iloc[-1] / daily_stats['Equity'].iloc[0]) ** (252/len(daily_stats)) - 1) * 100 if len(daily_stats) > 0 else 0
        max_dd = daily_stats['Drawdown'].min() * 100
        sharpe = (daily_stats['Returns'].mean() * 252) / (daily_stats['Returns'].std() * np.sqrt(252))
        avg_leverage = daily_stats['Leverage'].mean()
        
        # VISUALIZATION
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=(
                f"EQUITY CURVE (Log) | CAGR: {cagr:.1f}% | Sharpe: {sharpe:.2f}",
                f"UNDERWATER PLOT (Max DD: {max_dd:.1f}%)",
                f"LEVERAGE UTILIZATION (The Dimmer Switch) | Avg Lev: {avg_leverage:.2f}x",
                "ROLLING BETA (Correlation to Market)"
            )
        )

        # Row 1: Equity
        fig.add_trace(go.Scatter(x=daily_stats.index, y=daily_stats['Equity'], 
                                 name='Strategy', line=dict(color='#00ffcc', width=2)), row=1, col=1)
        if 'Bench_Equity' in daily_stats:
            fig.add_trace(go.Scatter(x=daily_stats.index, y=daily_stats['Bench_Equity'], 
                                     name='Benchmark', line=dict(color='#666', dash='dot')), row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1, gridcolor='#222')

        # Row 2: Drawdown (Red Area)
        fig.add_trace(go.Scatter(x=daily_stats.index, y=daily_stats['Drawdown'], 
                                 name='Drawdown', fill='tozeroy', line=dict(color='#ff5555', width=1)), row=2, col=1)

        # Row 3: Leverage (The Truth Serum)
        fig.add_trace(go.Scatter(x=daily_stats.index, y=daily_stats['Leverage'], 
                                 name='Gross Leverage', fill='tozeroy', line=dict(color='#ffaa00', width=1)), row=3, col=1)
        fig.add_hline(y=1.0, line_dash="dot", row=3, col=1, annotation_text="1.0x (Cash)")
        
        # Row 4: Beta
        fig.add_trace(go.Scatter(x=daily_stats.index, y=daily_stats['Beta'], 
                                 name='Rolling Beta', line=dict(color='#0088ff', width=1.5)), row=4, col=1)
        fig.add_hline(y=1.0, line_dash="dot", row=4, col=1)

        # STYLING
        fig.update_layout(
            template='plotly_dark',
            height=1400,
            title_text=f"ARCHITECT V11: FORENSIC AUDIT | Burn Rate: {burn_rate:.1f}%",
            hovermode="x unified",
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#111"
        )
        
        output_file = "architect_tearsheet_v11.html"
        fig.write_html(output_file)
        print(f"REPORT GENERATED: {output_file}")

if __name__ == "__main__":
    audit = ArchitectTearsheetV11(LOG_FILE)
    audit.generate_dashboard()
