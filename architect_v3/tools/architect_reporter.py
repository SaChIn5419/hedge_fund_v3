import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# --- CONFIGURATION ---
TRADE_LOG_PATH = 'trade_log.csv' 
BENCHMARK_TICKER = '^CRSLDX' # Nifty 500 (Use ^NSEI for Nifty 50 if 500 fails)
INITIAL_CAPITAL = 1000000

class ArchitectReportV8:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['date'] = pd.to_datetime(self.df['date']).dt.tz_localize(None)
        
    def fetch_benchmark(self):
        print(f"Fetching Benchmark Data ({BENCHMARK_TICKER})...")
        start_date = self.df['date'].min()
        end_date = self.df['date'].max()
        
        try:
            # Download and align
            # Use 'auto_adjust=True' to get adjusted close directly
            bench = yf.download(BENCHMARK_TICKER, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            if bench.empty:
                print("Nifty 500 failed. Falling back to Nifty 50 (^NSEI)...")
                bench = yf.download("^NSEI", start=start_date, end=end_date, progress=False, auto_adjust=True)

            if bench.empty:
                 return pd.Series(dtype=float)

            # Handle MultiIndex columns (common in new yfinance)
            if isinstance(bench.columns, pd.MultiIndex):
                # Try to find 'Close' in level 0
                if 'Close' in bench.columns.get_level_values(0):
                     bench = bench.xs('Close', axis=1, level=0)
                     # If multiple tickers (unlikely here), take first
                     if isinstance(bench, pd.DataFrame):
                          bench = bench.iloc[:, 0]
            elif 'Close' in bench.columns:
                 bench = bench['Close']
            
            bench = bench.rename('benchmark_close')
            return bench
        except Exception as e:
            print(f"Benchmark failed: {e}")
            return pd.Series(dtype=float)

    def generate_metrics(self):
        # 1. AGGREGATE STRATEGY PnL
        # Group by date to get Portfolio Level PnL
        daily_pnl = self.df.groupby('date')['net_pnl'].sum()
        portfolio = pd.DataFrame(daily_pnl).rename(columns={'net_pnl': 'pnl'})
        
        # Reconstruct Equity Curve
        portfolio['equity'] = INITIAL_CAPITAL + portfolio['pnl'].cumsum()
        portfolio['returns'] = portfolio['equity'].pct_change().fillna(0)
        
        # 2. MERGE BENCHMARK
        bench_series = self.fetch_benchmark()
        # Ensure index overlap
        if not bench_series.empty:
            # localize benchmark index to none to match portfolio
            bench_series.index = bench_series.index.tz_localize(None)
            portfolio = portfolio.join(bench_series, how='inner')
            
            # Normalize Benchmark to Strategy Base
            start_price = portfolio['benchmark_close'].iloc[0]
            portfolio['benchmark_equity'] = portfolio['benchmark_close'] * (INITIAL_CAPITAL / start_price)
            portfolio['bench_returns'] = portfolio['benchmark_close'].pct_change().fillna(0)
        else:
            portfolio['benchmark_equity'] = np.nan
            portfolio['bench_returns'] = 0.0
        
        # 3. ROLLING METRICS (6-Month Window)
        window = 126
        
        # Rolling Sharpe
        rolling_ret = portfolio['returns'].rolling(window).mean()
        rolling_std = portfolio['returns'].rolling(window).std()
        portfolio['rolling_sharpe'] = (rolling_ret / rolling_std) * np.sqrt(252)
        
        # Rolling Beta
        if 'bench_returns' in portfolio.columns:
            cov = portfolio['returns'].rolling(window).cov(portfolio['bench_returns'])
            var = portfolio['bench_returns'].rolling(window).var()
            portfolio['rolling_beta'] = cov / var
        
        # Drawdowns
        portfolio['peak'] = portfolio['equity'].cummax()
        portfolio['drawdown'] = (portfolio['equity'] - portfolio['peak']) / portfolio['peak']
        
        # Benchmark Drawdown
        if 'benchmark_equity' in portfolio.columns:
            bench_peak = portfolio['benchmark_equity'].cummax()
            portfolio['bench_drawdown'] = (portfolio['benchmark_equity'] - bench_peak) / bench_peak
        else:
             portfolio['bench_drawdown'] = 0.0

        # 4. REGIME GUARD VISUALIZATION
        # Extract the average system state per day (0 or 1)
        if 'market_regime' in self.df.columns:
            regime = self.df.groupby('date')['market_regime'].mean()
            portfolio = portfolio.join(regime)
        elif 'system_state' in self.df.columns:
            regime = self.df.groupby('date')['system_state'].mean()
            portfolio = portfolio.join(regime)
        else:
            portfolio['system_state'] = 1.0 # Default
            
        return portfolio

    def plot_dashboard(self, pf):
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=(
                "LOG-SCALE EQUITY: Architect (Green) vs Nifty 500 (Grey)", 
                "UNDERWATER PLOT: Drawdown Comparison", 
                "THE EDGE: Rolling Sharpe Ratio (6M)", 
                "THE BRAKE: Regime Guard Activation (Blue Area)"
            )
        )

        # ROW 1: EQUITY CURVE (Log Scale)
        fig.add_trace(go.Scatter(x=pf.index, y=pf['equity'], name='Architect V7',
                                 line=dict(color='#00ffcc', width=2)), row=1, col=1)
        if 'benchmark_equity' in pf.columns and not pf['benchmark_equity'].isna().all():
            fig.add_trace(go.Scatter(x=pf.index, y=pf['benchmark_equity'], name='Benchmark',
                                     line=dict(color='#666666', width=1, dash='dot')), row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1, gridcolor='#222')

        # ROW 2: DRAWDOWN (Red Area)
        fig.add_trace(go.Scatter(x=pf.index, y=pf['drawdown'], name='Strat Drawdown',
                                 fill='tozeroy', line=dict(color='#ff4444', width=1)), row=2, col=1)
        if 'bench_drawdown' in pf.columns and not pf['bench_drawdown'].isna().all():
            fig.add_trace(go.Scatter(x=pf.index, y=pf['bench_drawdown'], name='Bench Drawdown',
                                     line=dict(color='#666666', width=1, dash='dot')), row=2, col=1)
        
        # ROW 3: ROLLING SHARPE (Gold)
        fig.add_trace(go.Scatter(x=pf.index, y=pf['rolling_sharpe'], name='Rolling Sharpe',
                                 line=dict(color='#ffd700', width=1.5)), row=3, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="white", row=3, col=1)
        
        # ROW 4: REGIME STATE (The Switch)
        # Assuming column is 'system_state' or 'market_regime'
        regime_col = 'market_regime' if 'market_regime' in pf.columns else 'system_state'
        if regime_col in pf.columns:
             fig.add_trace(go.Scatter(x=pf.index, y=pf[regime_col], name='Regime State',
                                      fill='tozeroy', line=dict(color='#0088ff', width=1)), row=4, col=1)

        # STYLING
        fig.update_layout(
            template='plotly_dark',
            height=1200,
            title_text=f"ARCHITECT V8: QUANTUM DIAGNOSTICS | Final Equity: INR {pf['equity'].iloc[-1]:,.0f}",
            hovermode="x unified",
            paper_bgcolor="#050505",
            plot_bgcolor="#111111",
            font=dict(family="Courier New, monospace")
        )
        
        fig.write_html("architect_v8_dashboard.html")
        print("Dashboard Generated: architect_v8_dashboard.html")

# --- EXECUTE ---
if __name__ == "__main__":
    report = ArchitectReportV8(TRADE_LOG_PATH)
    portfolio_df = report.generate_metrics()
    report.plot_dashboard(portfolio_df)
