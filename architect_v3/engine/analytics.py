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
from scipy import stats

class PolarsTearSheet:
    """
    ARCHITECT V10: QUANTUM DIAGNOSTICS & TEARSHEET.
    Features: Advanced Metrics, Regime Analysis removal, Scatter Plots.
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
            pl.col("date").count().alias("days")
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
        
        # Ensure date is standard
        trades_df = trades_df.with_columns(pl.col("date").cast(pl.Date))

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
        """Fetches Benchmark Data."""
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

    def generate_v10_dashboard(self, df_pandas, bench_series):
        """Generates the Plotly Figure for V10 Dashboard."""
        
        # 1. Prepare Data
        combined = df_pandas.set_index('date').join(bench_series, how='left')
        
        # Fill Benchmark gaps
        combined['benchmark_close'] = combined['benchmark_close'].ffill()
        combined = combined.dropna(subset=['benchmark_close']) 

        if combined.empty:
             print("Warning: Combined data is empty after benchmark join.")
             return "<!-- No Data for Plot -->"

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
        
        # Rolling Alpha
        combined['rolling_alpha'] = (combined['ret'] - (combined['rolling_beta'] * combined['bench_returns']))
        combined['rolling_alpha_ann'] = combined['rolling_alpha'].rolling(window).mean() * 252

        # Rolling Z-Score
        r_mean = combined['ret'].rolling(window=20).mean()
        r_std = combined['ret'].rolling(window=20).std()
        combined['z_score'] = (combined['ret'] - r_mean) / r_std

        # 2. Create Subplots (5 Rows) - Swapped Regime for Returns Scatter logic
        fig = make_subplots(
            rows=5, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.3, 0.15, 0.15, 0.2, 0.2],
            specs=[[{"secondary_y": True}], [{}], [{}], [{"secondary_y": True}], [{}]],
            subplot_titles=(
                "EQUITY CURVE: Strategy (Green) vs Benchmark (Grey)", 
                "UNDERWATER PLOT: Drawdown Comparison", 
                "RETURN Z-SCORE (20-Day): Deviation from Mean",
                "CONVERGENCE: Rolling Alpha (Green) vs Beta (Orange)",
                "SCATTER: Strategy vs Benchmark Returns (Rolling Correlation)"
            )
        )

        # ROW 1: Equity
        fig.add_trace(go.Scatter(x=combined.index, y=combined['equity'], name='Strategy', 
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

        # ROW 4: Alpha/Beta
        fig.add_trace(go.Scatter(x=combined.index, y=combined['rolling_beta'], name='Beta (6m)', 
                                 line=dict(color='#ffa500', width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=combined.index, y=combined['rolling_alpha_ann'], name='Ann. Alpha', 
                                 line=dict(color='#00ffcc', width=1.5, dash='solid')), row=4, col=1, secondary_y=True)
        
        fig.update_yaxes(title="Beta", row=4, col=1, gridcolor='#222')
        fig.update_yaxes(title="Alpha", row=4, col=1, secondary_y=True, gridcolor='#222')

        # ROW 5: Rolling Correlation (Replcaing Regime)
        correlation = combined['ret'].rolling(60).corr(combined['bench_returns'])
        fig.add_trace(go.Scatter(x=combined.index, y=correlation, name='Roll Corr (60d)', 
                                 fill='tozeroy', line=dict(color='#0088ff')), row=5, col=1)
        fig.add_hline(y=0.0, line_color="#555", row=5, col=1)
        fig.update_yaxes(title="Corr", row=5, col=1, gridcolor='#222', range=[-1, 1])
        
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
        print("ARCHITECT: Compiling Hyper-Grid Tear Sheet (V10 Engine)...")
        
        # 1. DATA PREP
        df = df.with_columns([
            pl.col("equity").pct_change().alias("ret"),
            pl.col("equity").cum_max().alias("peak")
        ]).with_columns([
            (pl.col("equity") / pl.col("peak") - 1).alias("drawdown")
        ]).drop_nulls()

        # 2. METRICS CALCULATION
        initial_capital = df["equity"][0]
        final_equity = df["equity"][-1]
        cumulative_return = (final_equity / initial_capital) - 1
        
        total_days = df.height
        years = total_days / 252.0
        cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # Returns Analysis
        rets = df["ret"].to_numpy()
        avg_daily_ret = np.mean(rets)
        
        # Risk Ratios
        vol_ann = np.std(rets) * np.sqrt(252)
        sharpe = (np.mean(rets) * 252 - self.rf) / vol_ann if vol_ann > 0 else 0
        
        downside_rets = rets[rets < 0]
        downside_std = np.std(downside_rets) * np.sqrt(252)
        sortino = (np.mean(rets) * 252 - self.rf) / downside_std if downside_std > 0 else 0
        
        max_dd = df["drawdown"].min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Ulcer Index
        # SQRT(Mean(Drawdown^2))
        dd_sq = df["drawdown"].map_elements(lambda x: x**2, return_dtype=pl.Float64)
        ulcer_index = np.sqrt(dd_sq.mean())

        # Trading Physics
        days_won = np.sum(rets > 0)
        days_lost = np.sum(rets < 0)
        win_rate = days_won / total_days if total_days > 0 else 0
        
        avg_win = np.mean(rets[rets > 0]) if days_won > 0 else 0
        avg_loss = np.mean(rets[rets < 0]) if days_lost > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Kelly Criterion (Simple)
        # W - (1-W)/R where W=WinRate, R=Win/Loss Ratio
        kelly = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else 0
        
        # Gain/Pain Ratio
        # Sum(Positive Returns) / Abs(Sum(Negative Returns))
        sum_wins = np.sum(rets[rets > 0])
        sum_losses = abs(np.sum(rets[rets < 0]))
        gain_pain = sum_wins / sum_losses if sum_losses > 0 else 0

        # Tail Risk
        # 95% Historical VaR
        # Percentile logic: 5th percentile of returns
        var_95 = np.percentile(rets, 5) 
        
        # Expected Shortfall (cVaR) - Mean of returns below VaR
        cvar_95 = np.mean(rets[rets <= var_95])
        
        skewness = stats.skew(rets)
        kurtosis = stats.kurtosis(rets)

        # System Audit
        arith_pnl_sum = initial_capital * np.sum(rets)
        geometric_growth = final_equity - initial_capital
        
        # Volatility Drag = Geom - Arith (typically negative showing loss due to vol)
        vol_drag = geometric_growth - arith_pnl_sum

        # 3. DASHBOARD & PLOTS
        df_pd = df.to_pandas()
        df_pd['date'] = pd.to_datetime(df_pd['date'])
        
        bench = self.fetch_benchmark_data(df_pd['date'].min(), df_pd['date'].max())
        plotly_html = self.generate_v10_dashboard(df_pd, bench)

        # 4. TABLES
        eoy_table = self.get_eoy_table(df)
        top_winners, top_losers = self.get_top_trades(trades_df)
        dd_table = self.get_drawdown_table(df)

        # 5. HTML GENERATION
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ARCHITECT V10: Quantum Diagnostics</title>
            <style>
                body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0d0d0d; color: #e0e0e0; margin: 0; padding: 0; }}
                .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
                
                /* HEADER */
                h1 {{ color: #00ffcc; font-weight: 300; letter-spacing: 2px; text-transform: uppercase; border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 30px; }}
                h2 {{ color: #a0a0a0; font-size: 18px; margin-top: 30px; margin-bottom: 15px; border-left: 3px solid #00ffcc; padding-left: 10px; }}
                
                /* GRID LAYOUT */
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                
                /* METRIC TABLES */
                .metric-box {{ background: #161616; border: 1px solid #252525; padding: 15px; border-radius: 4px; }}
                .metric-title {{ color: #00ffcc; font-size: 14px; font-weight: bold; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #333; padding-bottom: 5px; }}
                
                .kv-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }}
                .kv-key {{ color: #888; }}
                .kv-val {{ font-family: 'Courier New', monospace; font-weight: bold; color: #ddd; }}
                .pos {{ color: #00ffcc; }} 
                .neg {{ color: #ff5555; }}
                
                /* DASHBOARD */
                .dashboard-container {{ background: #111; padding: 5px; border: 1px solid #222; margin-bottom: 30px; overflow: hidden; border-radius: 4px; }}
                
                /* DATA TABLES */
                .tables-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
                table {{ width: 100%; border-collapse: collapse; font-size: 12px; background: #111; }}
                th, td {{ padding: 10px; text-align: right; border-bottom: 1px solid #222; }}
                th {{ text-align: center; color: #888; text-transform: uppercase; font-weight: normal; font-size: 11px; }}
                td:first-child {{ text-align: left; color: #ccc; }}
                tr:hover {{ background-color: #1a1a1a; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Architect V10: Quantum Diagnostics</h1>
                
                <div class="metrics-grid">
                    <!-- COLUMN 1: SUMMARY -->
                    <div class="metric-box">
                        <div class="metric-title">Performance Summary</div>
                        <div class="kv-row"><span class="kv-key">Initial Capital</span><span class="kv-val">₹{initial_capital:,.2f}</span></div>
                        <div class="kv-row"><span class="kv-key">Final Equity</span><span class="kv-val">₹{final_equity:,.2f}</span></div>
                        <div class="kv-row"><span class="kv-key">Cumulative Return</span><span class="kv-val {'pos' if cumulative_return>0 else 'neg'}">{cumulative_return*100:,.2f}%</span></div>
                        <div class="kv-row"><span class="kv-key">CAGR</span><span class="kv-val {'pos' if cagr>0 else 'neg'}">{cagr*100:.2f}%</span></div>
                        <div class="kv-row"><span class="kv-key">Avg. Daily Return</span><span class="kv-val {'pos' if avg_daily_ret>0 else 'neg'}">{avg_daily_ret*100:.3f}%</span></div>
                    </div>
                    
                    <!-- COLUMN 2: RISK -->
                    <div class="metric-box">
                        <div class="metric-title">Risk & Ratios</div>
                        <div class="kv-row"><span class="kv-key">Volatility (Ann.)</span><span class="kv-val">{vol_ann*100:.2f}%</span></div>
                        <div class="kv-row"><span class="kv-key">Sharpe Ratio</span><span class="kv-val">{sharpe:.2f}</span></div>
                        <div class="kv-row"><span class="kv-key">Sortino Ratio</span><span class="kv-val">{sortino:.2f}</span></div>
                        <div class="kv-row"><span class="kv-key">Calmar Ratio</span><span class="kv-val">{calmar:.2f}</span></div>
                        <div class="kv-row"><span class="kv-key">Max Drawdown</span><span class="kv-val neg">{max_dd*100:.2f}%</span></div>
                        <div class="kv-row"><span class="kv-key">Ulcer Index</span><span class="kv-val">{ulcer_index:.2f}</span></div>
                    </div>

                    <!-- COLUMN 3: PHYSICS -->
                    <div class="metric-box">
                        <div class="metric-title">Trading Physics</div>
                        <div class="kv-row"><span class="kv-key">Kelly Criterion</span><span class="kv-val">{kelly*100:.2f}%</span></div>
                        <div class="kv-row"><span class="kv-key">Win Rate (Days)</span><span class="kv-val">{win_rate*100:.2f}%</span></div>
                        <div class="kv-row"><span class="kv-key">Avg. Win</span><span class="kv-val pos">{avg_win*100:.3f}%</span></div>
                        <div class="kv-row"><span class="kv-key">Avg. Loss</span><span class="kv-val neg">{avg_loss*100:.3f}%</span></div>
                        <div class="kv-row"><span class="kv-key">Win/Loss Ratio</span><span class="kv-val">{win_loss_ratio:.2f}</span></div>
                        <div class="kv-row"><span class="kv-key">Gain/Pain Ratio</span><span class="kv-val">{gain_pain:.2f}</span></div>
                    </div>

                    <!-- COLUMN 4: TAIL RISK -->
                    <div class="metric-box">
                        <div class="metric-title">Tail Risk / Distribution</div>
                        <div class="kv-row"><span class="kv-key">Daily VaR (95%)</span><span class="kv-val neg">{var_95*100:.2f}%</span></div>
                        <div class="kv-row"><span class="kv-key">Expected Shortfall</span><span class="kv-val neg">{cvar_95*100:.2f}%</span></div>
                        <div class="kv-row"><span class="kv-key">Skew</span><span class="kv-val">{skewness:.2f}</span></div>
                        <div class="kv-row"><span class="kv-key">Kurtosis</span><span class="kv-val">{kurtosis:.2f}</span></div>
                    </div>
                    
                    <!-- COLUMN 5: SYSTEM AUDIT -->
                    <div class="metric-box">
                        <div class="metric-title">System Audit</div>
                        <div class="kv-row"><span class="kv-key">Arithmetic PnL Sum</span><span class="kv-val">₹{arith_pnl_sum:,.0f}</span></div>
                        <div class="kv-row"><span class="kv-key">Geometric Growth</span><span class="kv-val">₹{geometric_growth:,.0f}</span></div>
                         <div class="kv-row"><span class="kv-key">Volatility Drag</span><span class="kv-val neg">₹{vol_drag:,.0f}</span></div>
                        <div class="kv-row"><span class="kv-key">Reality Check</span><span class="kv-val pos">COMPOUNDING</span></div>
                    </div>
                </div>

                <div class="dashboard-container">
                    {plotly_html}
                </div>

                <div class="tables-grid">
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

                <div class="tables-grid" style="margin-top: 20px;">
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

        output_file = "architect_tearsheet_v10.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"SUCCESS: V10 Tearsheet saved as '{output_file}'. Opening browser...")
        webbrowser.open('file://' + os.path.realpath(output_file))
