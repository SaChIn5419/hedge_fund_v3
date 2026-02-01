import polars as pl
import numpy as np
import json
import webbrowser
import os

class PolarsTearSheet:
    """
    ARCHITECT V3: Institutional Hyper-Grid Tear Sheet.
    With Gaussian Channel visualization and trade markers.
    """
    def __init__(self, risk_free_rate=0.06):
        self.rf = risk_free_rate

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

    def generate(self, df: pl.DataFrame, trades_df: pl.DataFrame = None):
        print("ARCHITECT: Compiling Hyper-Grid Tear Sheet with Trade Visualization...")
        
        # 1. CORE PHYSICS
        df = df.with_columns([
            pl.col("equity").pct_change().alias("ret"),
            pl.col("equity").cum_max().alias("peak")
        ]).with_columns([
            (pl.col("equity") / pl.col("peak") - 1).alias("drawdown")
        ]).drop_nulls()

        # 2. VECTORIZED METRICS
        total_days = df.height
        years = total_days / 252.0
        
        cum_ret = (df["equity"][-1] / df["equity"][0]) - 1
        cagr = (df["equity"][-1] / df["equity"][0]) ** (1 / years) - 1
        vol = df["ret"].std() * np.sqrt(252)
        sharpe = (df["ret"].mean() * 252 - self.rf) / vol
        
        neg_rets = df.filter(pl.col("ret") < 0)["ret"]
        pos_rets = df.filter(pl.col("ret") > 0)["ret"]
        sortino = (df["ret"].mean() * 252 - self.rf) / (neg_rets.std() * np.sqrt(252))
        max_dd = df["drawdown"].min()
        calmar = cagr / abs(max_dd)
        
        skew = df["ret"].skew()
        kurt = df["ret"].kurtosis()
        var_95 = df["ret"].quantile(0.05)
        cvar_95 = df.filter(pl.col("ret") <= var_95)["ret"].mean()
        
        win_rate = len(pos_rets) / total_days
        avg_win = pos_rets.mean()
        avg_loss = neg_rets.mean()
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        gain_pain = pos_rets.sum() / abs(neg_rets.sum())
        ulcer_index = np.sqrt((df["drawdown"]**2).mean())

        # 3. TABLES
        dd_table = self.get_drawdown_table(df)
        eoy_table = self.get_eoy_table(df)
        top_winners, top_losers = self.get_top_trades(trades_df)
        trade_freq = self.get_trade_frequency(trades_df)

        # 4. TRADE STATS
        total_trades = trades_df.height if trades_df is not None else 0
        unique_tickers = trades_df["ticker"].n_unique() if trades_df is not None else 0
        total_gross = trades_df["gross_pnl"].sum() if trades_df is not None else 0
        total_friction = trades_df["friction_pnl"].sum() if trades_df is not None else 0
        total_net = trades_df["net_pnl"].sum() if trades_df is not None else 0

        # 5. PREPARE JS DATA
        dates_js = json.dumps(df["date"].dt.strftime("%Y-%m-%d").to_list())
        equity_js = json.dumps(df["equity"].to_list())
        drawdown_js = json.dumps((df["drawdown"] * 100).to_list())
        
        # Daily P&L for histogram
        daily_pnl = (df["ret"] * 100).to_list()
        daily_pnl_js = json.dumps(daily_pnl)

        # 6. GENERATE HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ARCHITECT V3: Institutional Report</title>
            <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0a0a0a; color: #e0e0e0; margin: 30px; font-size: 13px; }}
                h1, h2, h3 {{ font-weight: 400; color: #00d4aa; border-bottom: 1px solid #333; padding-bottom: 5px; }}
                .grid-container {{ display: grid; grid-template-columns: 380px 1fr; gap: 20px; }}
                .metrics-table {{ width: 100%; border-collapse: collapse; background: #1a1a1a; box-shadow: 0 1px 3px rgba(0,0,0,0.3); }}
                .metrics-table td {{ padding: 6px 12px; border-bottom: 1px solid #2a2a2a; }}
                .metrics-table td:last-child {{ text-align: right; font-weight: 500; font-family: 'SF Mono', monospace; }}
                .section-header {{ background-color: #1e3a2f; font-weight: bold; text-align: left; padding: 8px 12px; color: #00d4aa; }}
                .negative {{ color: #ff6b6b; }} .positive {{ color: #00d4aa; }}
                .chart-box {{ background: #1a1a1a; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); border-radius: 4px; }}
                table.data-table {{ width: 100%; border-collapse: collapse; text-align: right; background: #1a1a1a; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); }}
                table.data-table th, table.data-table td {{ padding: 8px; border-bottom: 1px solid #2a2a2a; }}
                table.data-table th {{ background-color: #1e3a2f; text-align: center; color: #00d4aa; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }}
                .stat-card {{ background: #1a1a1a; padding: 15px; border-radius: 4px; text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; font-family: 'SF Mono', monospace; }}
                .stat-label {{ font-size: 11px; color: #888; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <h1>ARCHITECT V3: Quantitative Tear Sheet</h1>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value positive">₹{df['equity'][-1]:,.0f}</div>
                    <div class="stat-label">FINAL EQUITY</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value {'positive' if cagr>0 else 'negative'}">{cagr*100:.1f}%</div>
                    <div class="stat-label">CAGR</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{sharpe:.2f}</div>
                    <div class="stat-label">SHARPE RATIO</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value negative">{max_dd*100:.1f}%</div>
                    <div class="stat-label">MAX DRAWDOWN</div>
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{total_trades:,}</div>
                    <div class="stat-label">TOTAL TRADES</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{unique_tickers}</div>
                    <div class="stat-label">UNIQUE STOCKS</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value positive">₹{total_gross:,.0f}</div>
                    <div class="stat-label">GROSS P&L</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value negative">₹{total_friction:,.0f}</div>
                    <div class="stat-label">FRICTION PAID</div>
                </div>
            </div>
            
            <div class="grid-container">
                <div>
                    <table class="metrics-table">
                        <tr class="section-header"><td colspan="2">Returns & CAGR</td></tr>
                        <tr><td>Initial Capital</td><td>₹ {df['equity'][0]:,.2f}</td></tr>
                        <tr><td>Final Equity</td><td>₹ {df['equity'][-1]:,.2f}</td></tr>
                        <tr><td>Cumulative Return</td><td class="{'positive' if cum_ret>0 else 'negative'}">{cum_ret*100:.2f}%</td></tr>
                        <tr><td>CAGR</td><td class="{'positive' if cagr>0 else 'negative'}">{cagr*100:.2f}%</td></tr>
                        <tr><td>Avg. Daily Return</td><td>{df['ret'].mean()*100:.3f}%</td></tr>
                        
                        <tr class="section-header"><td colspan="2">Risk & Ratios</td></tr>
                        <tr><td>Volatility (Ann.)</td><td>{vol*100:.2f}%</td></tr>
                        <tr><td>Sharpe Ratio</td><td>{sharpe:.2f}</td></tr>
                        <tr><td>Sortino Ratio</td><td>{sortino:.2f}</td></tr>
                        <tr><td>Calmar Ratio</td><td>{calmar:.2f}</td></tr>
                        <tr><td>Max Drawdown</td><td class="negative">{max_dd*100:.2f}%</td></tr>
                        <tr><td>Ulcer Index</td><td>{ulcer_index:.2f}</td></tr>

                        <tr class="section-header"><td colspan="2">Trading Physics</td></tr>
                        <tr><td>Kelly Criterion</td><td>{kelly*100:.2f}%</td></tr>
                        <tr><td>Win Rate (Days)</td><td>{win_rate*100:.2f}%</td></tr>
                        <tr><td>Avg. Win</td><td class="positive">{avg_win*100:.3f}%</td></tr>
                        <tr><td>Avg. Loss</td><td class="negative">{avg_loss*100:.3f}%</td></tr>
                        <tr><td>Win/Loss Ratio</td><td>{win_loss_ratio:.2f}</td></tr>
                        <tr><td>Gain/Pain Ratio</td><td>{gain_pain:.2f}</td></tr>

                        <tr class="section-header"><td colspan="2">Tail Risk / Distribution</td></tr>
                        <tr><td>Daily VaR (95%)</td><td class="negative">{var_95*100:.2f}%</td></tr>
                        <tr><td>Expected Shortfall (cVaR)</td><td class="negative">{cvar_95*100:.2f}%</td></tr>
                        <tr><td>Skew</td><td>{skew:.2f}</td></tr>
                        <tr><td>Kurtosis</td><td>{kurt:.2f}</td></tr>
                    </table>
                </div>

                <div>
                    <div class="chart-box" id="equity_plot" style="height: 350px;"></div>
                    <div class="chart-box" id="dd_plot" style="height: 200px;"></div>
                    <div class="chart-box" id="returns_hist" style="height: 200px;"></div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <h3>EOY Returns</h3>
                            <table class="data-table">
                                <thead><tr><th>Year</th><th>Strategy</th></tr></thead>
                                <tbody>
                                    {"".join([f"<tr><td style='text-align:center;'>{row['Year']}</td><td class='{'positive' if row['Strategy']>0 else 'negative'}'>{row['Strategy']*100:.2f}%</td></tr>" for row in eoy_table])}
                                </tbody>
                            </table>
                        </div>

                        <div>
                            <h3>Top 20 Stocks by P&L</h3>
                            <table class="data-table">
                                <thead><tr><th>Ticker</th><th>Trades</th><th>Total P&L</th></tr></thead>
                                <tbody>
                                    {"".join([f"<tr><td style='text-align:left;'>{r['ticker']}</td><td style='text-align:center;'>{r['trades']}</td><td class='{'positive' if r['total_pnl']>0 else 'negative'}'>₹{r['total_pnl']:,.0f}</td></tr>" for r in trade_freq])}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
                        <div>
                            <h3>Top 20 Winning Trades</h3>
                            <table class="data-table">
                                <thead><tr><th>Date</th><th>Ticker</th><th>P&L</th></tr></thead>
                                <tbody>
                                    {"".join([f"<tr><td style='text-align:center;'>{r['date']}</td><td style='text-align:left;'>{r['ticker']}</td><td class='positive'>₹{r['net_pnl']:,.0f}</td></tr>" for r in top_winners])}
                                </tbody>
                            </table>
                        </div>

                        <div>
                            <h3>Top 20 Losing Trades</h3>
                            <table class="data-table">
                                <thead><tr><th>Date</th><th>Ticker</th><th>P&L</th></tr></thead>
                                <tbody>
                                    {"".join([f"<tr><td style='text-align:center;'>{r['date']}</td><td style='text-align:left;'>{r['ticker']}</td><td class='negative'>₹{r['net_pnl']:,.0f}</td></tr>" for r in top_losers])}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <h3>Worst 10 Drawdowns</h3>
                    <table class="data-table">
                        <thead><tr><th>Started</th><th>Recovered</th><th>Drawdown</th><th>Days</th></tr></thead>
                        <tbody>
                            {"".join([f"<tr><td style='text-align:center;'>{r['started']}</td><td style='text-align:center;'>{r['recovered']}</td><td class='negative'>{r['drawdown']*100:.2f}%</td><td style='text-align:center;'>{r['days']}</td></tr>" for r in dd_table])}
                        </tbody>
                    </table>
                </div>
            </div>

            <script>
                var layout_dark = {{
                    paper_bgcolor: '#1a1a1a', plot_bgcolor: '#1a1a1a',
                    font: {{color: '#e0e0e0'}},
                    xaxis: {{gridcolor: '#2a2a2a'}},
                    yaxis: {{gridcolor: '#2a2a2a'}}
                }};

                Plotly.newPlot('equity_plot', [{{
                    x: {dates_js}, y: {equity_js}, type: 'scatter', mode: 'lines',
                    line: {{color: '#00d4aa', width: 2}}, fill: 'tozeroy', fillcolor: 'rgba(0, 212, 170, 0.1)',
                    name: 'Equity'
                }}], {{
                    ...layout_dark,
                    title: {{text: 'Equity Curve (Gaussian Channel Strategy)', font: {{color: '#00d4aa'}}}},
                    margin: {{t: 40, b: 30, l: 60, r: 20}}
                }});

                Plotly.newPlot('dd_plot', [{{
                    x: {dates_js}, y: {drawdown_js}, type: 'scatter', mode: 'lines',
                    line: {{color: '#ff6b6b', width: 1}}, fill: 'tozeroy', fillcolor: 'rgba(255, 107, 107, 0.3)',
                    name: 'Drawdown'
                }}], {{
                    ...layout_dark,
                    title: {{text: 'Drawdowns (%)', font: {{color: '#ff6b6b'}}}},
                    margin: {{t: 40, b: 30, l: 60, r: 20}}
                }});

                Plotly.newPlot('returns_hist', [{{
                    x: {daily_pnl_js}, type: 'histogram', nbinsx: 100,
                    marker: {{color: '#00d4aa', line: {{color: '#0a0a0a', width: 0.5}}}},
                    name: 'Daily Returns'
                }}], {{
                    ...layout_dark,
                    title: {{text: 'Daily Returns Distribution (%)', font: {{color: '#00d4aa'}}}},
                    margin: {{t: 40, b: 30, l: 60, r: 20}},
                    bargap: 0.05
                }});
            </script>
        </body>
        </html>
        """

        output_file = "architect_tearsheet.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"SUCCESS: Report saved as '{output_file}'. Opening browser...")
        webbrowser.open('file://' + os.path.realpath(output_file))
