import polars as pl
import pandas as pd
import numpy as np
import json
import webbrowser
import os
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

class PolarsTearSheet:
    """
    ARCHITECT V9: QUANTUM DIAGNOSTICS & TEARSHEET.
    Features: Rolling Z-Score, Alpha/Beta Convergence, Special Trading Days support.
    """
    def __init__(self, risk_free_rate=0.06, benchmark_ticker="^NSEI"):
        self.rf = risk_free_rate
        self.benchmark_ticker = benchmark_ticker

    def get_drawdown_table(self, df):
        """Identifies the Worst 10 Drawdowns."""
        dd_df = df.with_columns([
            pl.col("equity").cum_max().alias("peak")
        ]).with_columns([
            (pl.col("equity") / pl.col("peak") - 1).alias("drawdown"),
            (pl.col("equity") == pl.col("equity").cum_max()).cast(pl.Int32).cum_sum().alias("dd_regime")
        ])

        top_10 = dd_df.filter(pl.col("drawdown") < 0).group_by("dd_regime").agg([
            pl.col("drawdown").min().alias("drawdown"),
            pl.col("date").first().alias("started"),
            pl.col("date").last().alias("recovered"),
            pl.count().alias("days")
        ]).sort("drawdown").head(10).drop("dd_regime")
        
        return top_10.with_columns([
            pl.col("started").dt.strftime("%Y-%m-%d"),
            pl.col("recovered").dt.strftime("%Y-%m-%d")
        ]).to_dicts()

    def get_eoy_table(self, df):
        """Calculates End-of-Year (EOY) Returns."""
        eoy = df.with_columns([
            pl.col("date").dt.year().alias("Year")
        ]).group_by("Year").agg([
            (((1 + pl.col("ret")).product()) - 1).alias("Strategy")
        ]).sort("Year")
        return eoy.to_dicts()

    def get_top_trades(self, trades_df, n=20):
        """Get top N profitable and worst N trades."""
        if trades_df is None or trades_df.height == 0:
            return [], []
        
        top_winners = trades_df.sort("net_pnl", descending=True).head(n).select([
            "date", "ticker", "close", "net_pnl"
        ]).with_columns([
            pl.col("date").dt.strftime("%Y-%m-%d")
        ]).to_dicts()
        
        top_losers = trades_df.sort("net_pnl").head(n).select([
            "date", "ticker", "close", "net_pnl"
        ]).with_columns([
            pl.col("date").dt.strftime("%Y-%m-%d")
        ]).to_dicts()
        
        return top_winners, top_losers

    def get_trade_frequency(self, trades_df):
        """Get trade count by ticker."""
        if trades_df is None:
            return []
        freq = trades_df.group_by("ticker").agg([
            pl.count().alias("trades"),
            pl.col("net_pnl").sum().alias("total_pnl")
        ]).sort("total_pnl", descending=True).head(20).to_dicts()
        return freq
    
    def fetch_benchmark_data(self, start_date, end_date):
        """Fetches Benchmark Data for V9 Comparison."""
        print(f"ARCHITECT: Fetching Benchmark ({self.benchmark_ticker})...")
        try:
            # Download with auto_adjust to get proper Close prices
            bench = yf.download(self.benchmark_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            if bench.empty:
                print("ARCHITECT WARN: Benchmark download failed (Empty).")
                return None

            # Handle MultiIndex if present
            if isinstance(bench.columns, pd.MultiIndex):
                if 'Close' in bench.columns.get_level_values(0):
                     bench = bench.xs('Close', axis=1, level=0)
                     if isinstance(bench, pd.DataFrame):
                          bench = bench.iloc[:, 0]
            elif 'Close' in bench.columns:
                 bench = bench['Close']
            
            bench = bench.rename('benchmark_close')
            bench.index = bench.index.tz_localize(None) # Ensure naive datetime
            return bench
        except Exception as e:
            print(f"ARCHITECT WARN: Benchmark download error ({e})")
            return None

    def generate_v9_dashboard(self, df_pandas, bench_series):
        """Generates the Plotly Figure for V9 Dashboard (Z-Score & Convergence)."""
        
        # 1. Prepare Data
        # Align Benchmark
        combined = df_pandas.set_index('date').join(bench_series, how='left')
        
        # Fill Benchmark gaps (ffill) - Crucial for holidays/missing days
        combined['benchmark_close'] = combined['benchmark_close'].ffill()
        combined = combined.dropna(subset=['benchmark_close']) # Drop start if benchmark missing

        # Calculate Benchmark Equity
        initial_equity = combined['equity'].iloc[0]
        start_price = combined['benchmark_close'].iloc[0]
        combined['benchmark_equity'] = combined['benchmark_close'] * (initial_equity / start_price)
        combined['bench_returns'] = combined['benchmark_close'].pct_change().fillna(0)
        combined['bench_drawdown'] = (combined['benchmark_equity'] - combined['benchmark_equity'].cummax()) / combined['benchmark_equity'].cummax()

        # Rolling Metrics (126 Days / 6 Months)
        window = 126
        
        # Rolling Beta
        rolling_cov = combined['ret'].rolling(window).cov(combined['bench_returns'])
        rolling_var = combined['bench_returns'].rolling(window).var()
        combined['rolling_beta'] = rolling_cov / rolling_var
        
        # Rolling Alpha (Jensen's)
        # Annualized Alpha = (RollingRet - Beta * RollingBenchRet) * 252 (simplified)
        # Or just daily alpha spread
        combined['rolling_alpha'] = (combined['ret'] - (combined['rolling_beta'] * combined['bench_returns']))
        combined['rolling_alpha_ann'] = combined['rolling_alpha'].rolling(window).mean() * 252

        # Rolling Z-Score (of Strategy Returns)
        # Z = (Return - RollingMean) / RollingStd
        r_mean = combined['ret'].rolling(window=20).mean()
        r_std = combined['ret'].rolling(window=20).std()
        combined['z_score'] = (combined['ret'] - r_mean) / r_std

        # 2. Create Subplots (5 Rows)
        fig = make_subplots(
            rows=5, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.3, 0.15, 0.15, 0.2, 0.2],
            specs=[[{"secondary_y": True}], [{}], [{}], [{"secondary_y": True}], [{}]],
            subplot_titles=(
                "EQUITY CURVE: Strategy (Green) vs Nifty 50 (Grey)", 
                "UNDERWATER PLOT: Drawdown Comparison", 
                "RETURN Z-SCORE (20-Day): Deviation from Mean",
                "CONVERGENCE: Rolling Alpha (Green) vs Beta (Orange)",
                "REGIME GUARD: Market State (Blue = ON)"
            )
        )

        # ROW 1: Equity
        fig.add_trace(go.Scatter(x=combined.index, y=combined['equity'], name='Architect V7', 
                                 line=dict(color='#00ffcc', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=combined.index, y=combined['benchmark_equity'], name='Benchmark', 
                                 line=dict(color='#666666', dash='dot', width=1)), row=1, col=1)
        fig.update_yaxes(type="log", title="Log Equity", row=1, col=1, gridcolor='#222')

        # ROW 2: Drawdown
        fig.add_trace(go.Scatter(x=combined.index, y=combined['drawdown'], name='Strat DD', 
                                 fill='tozeroy', line=dict(color='#ff5555', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=combined.index, y=combined['bench_drawdown'], name='Bench DD', 
                                 line=dict(color='#666666', dash='dot', width=1)), row=2, col=1)
        fig.update_yaxes(title="DD %", row=2, col=1, gridcolor='#222')

        # ROW 3: Rolling Z-Score
        fig.add_trace(go.Scatter(x=combined.index, y=combined['z_score'], name='Z-Score', 
                                 line=dict(color='#b388ff', width=1)), row=3, col=1)
        fig.add_hline(y=2.0, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=-2.0, line_dash="dot", line_color="red", row=3, col=1)
        fig.update_yaxes(title="Sigma", row=3, col=1, gridcolor='#222')

        # ROW 4: Alpha/Beta Convergence (Dual Axis)
        fig.add_trace(go.Scatter(x=combined.index, y=combined['rolling_beta'], name='Beta (6m)', 
                                 line=dict(color='#ffa500', width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=combined.index, y=combined['rolling_alpha_ann'], name='Ann. Alpha', 
                                 line=dict(color='#00ffcc', width=1.5, dash='solid')), row=4, col=1, secondary_y=True)
        
        fig.update_yaxes(title="Beta", row=4, col=1, gridcolor='#222')
        fig.update_yaxes(title="Alpha", row=4, col=1, secondary_y=True, gridcolor='#222')

        # ROW 5: Regime State
        if 'market_regime' in combined.columns:
            fig.add_trace(go.Scatter(x=combined.index, y=combined['market_regime'], name='Regime', 
                                     fill='tozeroy', line=dict(color='#0088ff')), row=5, col=1)
        else:
             fig.add_trace(go.Scatter(x=combined.index, y=[1]*len(combined), name='Regime (Unknown)', 
                                      line=dict(color='#333')), row=5, col=1)
        
        fig.update_layout(
            height=1400, 
            template='plotly_dark', 
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#111111",
            font=dict(family="Courier New, monospace"),
            hovermode="x unified",
            margin=dict(l=50, r=50, t=60, b=40)
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def generate(self, df: pl.DataFrame, trades_df: pl.DataFrame = None):
        print("ARCHITECT: Compiling Hyper-Grid Tear Sheet (V9 Engine)...")
        
        # 1. CORE PHYSICS (Polars)
        df = df.with_columns([
            pl.col("equity").pct_change().alias("ret"),
            pl.col("equity").cum_max().alias("peak")
        ]).with_columns([
            (pl.col("equity") / pl.col("peak") - 1).alias("drawdown")
        ]).drop_nulls()

        # 2. METRICS
        total_days = df.height
        years = total_days / 252.0
        cum_ret = (df["equity"][-1] / df["equity"][0]) - 1
        cagr = (df["equity"][-1] / df["equity"][0]) ** (1 / years) - 1
        vol = df["ret"].std() * np.sqrt(252)
        sharpe = (df["ret"].mean() * 252 - self.rf) / vol
        max_dd = df["drawdown"].min()
        
        # 3. PREPARE V9 DASHBOARD
        df_pd = df.to_pandas()
        df_pd['date'] = pd.to_datetime(df_pd['date'])
        
        if trades_df is not None:
             daily_regime = trades_df.group_by("date").agg(pl.col("market_regime").mean()).sort("date").to_pandas()
             daily_regime['date'] = pd.to_datetime(daily_regime['date'])
             df_pd = df_pd.merge(daily_regime, on='date', how='left').fillna(1.0)

        bench = self.fetch_benchmark_data(df_pd['date'].min(), df_pd['date'].max())
        plotly_html = self.generate_v9_dashboard(df_pd, bench)

        # 4. PREPARE TABLES
        eoy_table = self.get_eoy_table(df)
        top_winners, top_losers = self.get_top_trades(trades_df)
        trade_freq = self.get_trade_frequency(trades_df)
        dd_table = self.get_drawdown_table(df)
        
        total_trades = trades_df.height if trades_df is not None else 0
        total_net = trades_df["net_pnl"].sum() if trades_df is not None else 0
        total_friction = trades_df["friction_pnl"].sum() if trades_df is not None else 0

        # 5. GENERATE HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ARCHITECT V9: Quantum Diagnostics</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; background-color: #050505; color: #e0e0e0; margin: 0; padding: 20px; }}
                h1, h2 {{ color: #00ffcc; font-weight: 300; letter-spacing: 2px; text-transform: uppercase; border-bottom: 1px solid #333; }}
                .container {{ max-width: 1400px; margin: 0 auto; }}
                .grid-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
                .card {{ background: #111; padding: 20px; border: 1px solid #222; text-align: center; }}
                .value {{ font-size: 28px; font-weight: bold; font-family: 'Courier New', monospace; }}
                .label {{ font-size: 12px; color: #888; margin-top: 5px; text-transform: uppercase; }}
                .pos {{ color: #00ffcc; }} .neg {{ color: #ff5555; }}
                
                .dashboard-container {{ background: #111; padding: 10px; border: 1px solid #222; margin-bottom: 30px; }}
                
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 12px; background: #111; }}
                th, td {{ padding: 10px; text-align: right; border-bottom: 1px solid #222; }}
                th {{ text-align: center; color: #00ffcc; text-transform: uppercase; border-bottom: 2px solid #00ffcc; }}
                td:first-child {{ text-align: left; color: #fff; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Architect V9: Quantum Diagnostics</h1>
                
                <div class="grid-stats">
                    <div class="card"><div class="value pos">₹{df['equity'][-1]:,.0f}</div><div class="label">Final Equity</div></div>
                    <div class="card"><div class="value {'pos' if cagr>0 else 'neg'}">{cagr*100:.2f}%</div><div class="label">CAGR</div></div>
                    <div class="card"><div class="value">{sharpe:.2f}</div><div class="label">Sharpe Ratio</div></div>
                    <div class="card"><div class="value neg">{max_dd*100:.2f}%</div><div class="label">Max Drawdown</div></div>
                    <div class="card"><div class="value">{total_trades}</div><div class="label">Total Trades</div></div>
                    <div class="card"><div class="value neg">₹{total_friction:,.0f}</div><div class="label">Friction Paid</div></div>
                </div>

                <div class="dashboard-container">
                    {plotly_html}
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                    <div>
                        <h2>EOY Returns</h2>
                        <table>
                            <thead><tr><th>Year</th><th>Return</th></tr></thead>
                            <tbody>
                                {"".join([f"<tr><td>{r['Year']}</td><td class='{'pos' if r['Strategy']>0 else 'neg'}'>{r['Strategy']*100:.2f}%</td></tr>" for r in eoy_table])}
                            </tbody>
                        </table>
                    </div>
                    <div>
                        <h2>Worst Drawdowns</h2>
                        <table>
                            <thead><tr><th>Started</th><th>Recovered</th><th>Depth</th><th>Days</th></tr></thead>
                            <tbody>
                                {"".join([f"<tr><td>{r['started']}</td><td>{r['recovered']}</td><td class='neg'>{r['drawdown']*100:.2f}%</td><td>{r['days']}</td></tr>" for r in dd_table])}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                     <div>
                        <h2>Top Winners</h2>
                        <table>
                            <thead><tr><th>Date</th><th>Ticker</th><th>Profit</th></tr></thead>
                            <tbody>
                                {"".join([f"<tr><td>{r['date']}</td><td>{r['ticker']}</td><td class='pos'>₹{r['net_pnl']:,.0f}</td></tr>" for r in top_winners])}
                            </tbody>
                        </table>
                    </div>
                    <div>
                        <h2>Top Losers</h2>
                         <table>
                            <thead><tr><th>Date</th><th>Ticker</th><th>Loss</th></tr></thead>
                            <tbody>
                                {"".join([f"<tr><td>{r['date']}</td><td>{r['ticker']}</td><td class='neg'>₹{r['net_pnl']:,.0f}</td></tr>" for r in top_losers])}
                            </tbody>
                        </table>
                    </div>
                </div>

            </div>
        </body>
        </html>
        """

        output_file = "architect_tearsheet_v3.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"SUCCESS: V9 Tearsheet saved as '{output_file}'. Opening browser...")
        webbrowser.open('file://' + os.path.realpath(output_file))
