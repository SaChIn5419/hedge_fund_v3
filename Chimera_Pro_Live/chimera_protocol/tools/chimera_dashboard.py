import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import os

# --- PATH CONFIG ---
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "trade_log.csv")
BENCHMARK = '^CRSLDX' 
SAFE_HAVEN = 'GOLDBEES.NS'

def get_current_holdings():
    """Reads the last day's holdings from the trade log."""
    print(f"Reading Holdings from: {LOG_FILE}")
    try:
        df = pd.read_csv(LOG_FILE)
        # Parse Dates
        df['date'] = pd.to_datetime(df['date'])
        
        # Get Last Date
        last_date = df['date'].max()
        print(f"Latest Portfolio Date: {last_date.date()}")
        
        # Filter for Last Date and Positive Weight
        current = df[(df['date'] == last_date) & (df['weight'] > 0)]
        
        tickers = current['ticker'].unique().tolist()
        print(f"Active Positions ({len(tickers)}): {tickers}")
        
        if not tickers:
            print("WARNING: Portfolio is 100% Cash. Using Watchlist defaults.")
            return ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'] # Default Fallback
            
        return tickers
        
    except Exception as e:
        print(f"ERROR reading trade log: {e}")
        return ['RELIANCE.NS'] # Fallback

# --- DYNAMIC CONFIG ---
TICKERS = get_current_holdings()

class ChimeraCommandCenter:
    def __init__(self):
        print("--- INITIALIZING CHIMERA PRIME DASHBOARD ---")
        self.data = {}
        self.tickers = TICKERS
        
    def fetch_live_snapshot(self):
        print("Fetching Live Data Grid...")
        # Fetch last 60 days for metrics
        # Ensure unique and cleaned list
        symbol_list = list(set(self.tickers + [BENCHMARK, SAFE_HAVEN]))
        
        try:
            df = yf.download(symbol_list, period="60d", progress=False)
            
            # Post-Processing for yfinance MultiIndex
            # Case 1: Multiple Tickers (MultiIndex Columns)
            if isinstance(df.columns, pd.MultiIndex):
                # Dropping the level logic to be robust
                # We want a MultiIndex of [Date, Ticker] or similar, but the user's stacking logic is tricky.
                # Let's try the user's logic:
                # df.stack(level=1) stacks the Ticker level (level 1) to Index.
                # Columns become OHLC.
                # This works if columns are (PriceType, Ticker).
                
                # Check levels
                if df.columns.nlevels == 2:
                    # Level 0 = Price Type (Close, Open...), Level 1 = Ticker
                    # Stack level 1 to get Ticker as index
                    try:
                        df_stacked = df.stack(level=1, future_stack=True)
                    except:
                        df_stacked = df.stack(level=1)
                    
                    df_stacked.index.names = ['Date', 'Ticker']
                    df = df_stacked.reset_index()
                
            # Case 2: Single Ticker (Result is just OHLC)
            else:
                # Add Ticker Column manually
                df['Ticker'] = symbol_list[0]
                df = df.reset_index()
                
            self.data = df
            return df
            
        except Exception as e:
            print(f"Data Fetch Error: {e}")
            return pd.DataFrame()

    def calculate_engine_metrics(self):
        """
        Calculates the Status of every Asset in the Universe.
        """
        engine_grid = []
        
        # Standardize Columns
        if 'Close' not in self.data.columns:
            print("CRITICAL: No Close price data.")
            return pd.DataFrame()
            
        # Process each ticker
        for ticker in self.tickers:
            # Flexible filtering
            if 'Ticker' in self.data.columns:
                stock_df = self.data[self.data['Ticker'] == ticker].copy()
            else:
                stock_df = self.data.copy() # Only one ticker case
                
            if stock_df.empty: continue
            
            # Ensure sorting
            stock_df = stock_df.sort_values('Date')
            
            # 1. Price Physics
            try:
                last_price = stock_df['Close'].iloc[-1]
                prev_price = stock_df['Close'].iloc[-2]
                change = (last_price - prev_price) / prev_price
                
                # 2. Gaussian Signal (Simplified)
                mean = stock_df['Close'].rolling(20).mean().iloc[-1]
                std = stock_df['Close'].rolling(20).std().iloc[-1]
                upper = mean + (2*std)
                lower = mean - (2*std)
                
                if last_price > upper: signal = "BREAKOUT"
                elif last_price < lower: signal = "BREAKDOWN"
                else: signal = "NEUTRAL"
                
                # 3. Energy (Slope)
                # Simple slope proxy: 5-day ROC
                energy_score = (stock_df['Close'].iloc[-1] / stock_df['Close'].iloc[-6]) - 1
                
                # 4. Volume Profile Status (Proxy)
                # If price is near 20-day mean, assume "IN VALUE" (Chop)
                # If price is > 1 std dev away, assume "VACUUM" (Trend)
                z_score = (last_price - mean) / std if std != 0 else 0
                if abs(z_score) < 1.0:
                    vp_status = "TRAPPED (CHOP)"
                else:
                    vp_status = "VACUUM (RUN)"
                    
                engine_grid.append({
                    'Ticker': ticker,
                    'Price': round(last_price, 2),
                    'Change %': round(change * 100, 2),
                    'Signal': signal,
                    'Energy': round(energy_score * 100, 2),
                    'Structure': vp_status
                })
            except Exception as e:
                print(f"Metric Error for {ticker}: {e}")
                continue
            
        return pd.DataFrame(engine_grid)

    def calculate_system_health(self):
        """
        Checks Regime, Volatility, and Correlations.
        """
        # 1. Regime Check
        if 'Ticker' in self.data.columns:
            bench = self.data[self.data['Ticker'] == BENCHMARK].copy()
        else:
             bench = pd.DataFrame() # Should not happen if multi-download
        
        if bench.empty:
            # Fallback if benchmark download failed
            regime = "UNKNOWN"
            current_vol = 0
            price = 0
            sma200 = 0
        else:
            bench = bench.sort_values('Date')
            sma200 = bench['Close'].rolling(50).mean().iloc[-1] 
            price = bench['Close'].iloc[-1]
            regime = "BULLISH" if price > sma200 else "BEARISH"
            
            # 2. Volatility Temperature
            returns = bench['Close'].pct_change()
            current_vol = returns.tail(20).std() * np.sqrt(252) * 100
        
        # 3. Correlation Matrix (Diversity Check)
        if 'Ticker' in self.data.columns:
            pivot = self.data[self.data['Ticker'].isin(self.tickers)].pivot(index='Date', columns='Ticker', values='Close')
            corr_matrix = pivot.pct_change().corr()
            avg_corr = corr_matrix.mean().mean()
        else:
            avg_corr = 1.0
        
        return {
            'Regime': regime,
            'VIX_Proxy': round(current_vol, 2),
            'Avg_Corr': round(avg_corr, 2),
            'Data_Lag': 0 # Real-time ish
        }

    def generate_dashboard(self):
        self.fetch_live_snapshot()
        grid = self.calculate_engine_metrics()
        health = self.calculate_system_health()
        
        if grid.empty:
            print("Dashboard Generation Aborted: No Data.")
            return

        # --- BUILD THE DASHBOARD ---
        fig = make_subplots(
            rows=3, cols=3,
            column_widths=[0.3, 0.4, 0.3],
            row_heights=[0.2, 0.4, 0.4],
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "table", "colspan": 2}, None, {"type": "xy"}], 
                [{"type": "heatmap", "colspan": 2}, None, {"type": "indicator"}] 
            ],
            subplot_titles=("MARKET REGIME", "VOLATILITY TEMP", "DIVERSITY SCORE", 
                            "ACTIVE SIGNAL GRID", "BENCHMARK TREND", 
                            "CORRELATION MATRIX", "SYSTEM LATENCY")
        )
        
        # 1. HUD: REGIME
        color = "#00ffcc" if health['Regime'] == "BULLISH" else "#ff5555"
        fig.add_trace(go.Indicator(
            mode = "number+gauge", value = 1 if health['Regime']=="BULLISH" else 0,
            title = {"text": f"REGIME: {health['Regime']}"},
            gauge = {'axis': {'range': [0, 1]}, 'bar': {'color': color}},
            number = {'font': {'color': color}}
        ), row=1, col=1)

        # 2. HUD: VOLATILITY GAUGE
        fig.add_trace(go.Indicator(
            mode = "number+gauge", value = health['VIX_Proxy'],
            title = {"text": "VOLATILITY %"},
            gauge = {
                'axis': {'range': [0, 50]},
                'bar': {'color': "#ffaa00"},
                'steps': [
                    {'range': [0, 15], 'color': "#1e3a2f"},
                    {'range': [15, 25], 'color': "#3a3a1e"},
                    {'range': [25, 50], 'color': "#3a1e1e"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 25}
            }
        ), row=1, col=2)
        
        # 3. HUD: DIVERSITY (Avg Correlation)
        fig.add_trace(go.Indicator(
            mode = "number", value = health['Avg_Corr'],
            title = {"text": "AVG CORRELATION"},
            number = {'suffix': " rho"}
        ), row=1, col=3)

        # 4. ENGINE ROOM: SIGNAL GRID (Table)
        # Color coding rows based on Signal
        cell_colors = []
        for sig in grid['Signal']:
            if sig == "BREAKOUT": cell_colors.append("#1e3a2f") # Green tint
            elif sig == "BREAKDOWN": cell_colors.append("#3a1e1e") # Red tint
            else: cell_colors.append("#1a1a1a") # Dark
            
        fig.add_trace(go.Table(
            header=dict(values=list(grid.columns),
                        fill_color='#333',
                        align='left', font=dict(color='white', size=12)),
            cells=dict(values=[grid[k].tolist() for k in grid.columns],
                       fill_color=[cell_colors * len(grid.columns)], # Apply row colors
                       align='left', font=dict(color='lightgrey', size=11))
        ), row=2, col=1)

        # 5. MINI CHART: BENCHMARK
        if 'Ticker' in self.data.columns:
            bench_df = self.data[self.data['Ticker'] == BENCHMARK]
            if not bench_df.empty:
                bench_df = bench_df.sort_values('Date')
                fig.add_trace(go.Scatter(x=bench_df['Date'], y=bench_df['Close'], 
                                        line=dict(color='#0088ff', width=2), name="Benchmark"), row=2, col=3)

        # 6. RISK MATRIX: CORRELATION HEATMAP
        if 'Ticker' in self.data.columns:
            pivot = self.data[self.data['Ticker'].isin(self.tickers)].pivot(index='Date', columns='Ticker', values='Close')
            corr = pivot.pct_change().corr()
            
            fig.add_trace(go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale='Viridis',
                zmin=-1, zmax=1
            ), row=3, col=1)
        
        # 7. SYSTEM HEALTH: LATENCY
        fig.add_trace(go.Indicator(
            mode = "number", value = health['Data_Lag'],
            title = {"text": "DATA LAG (Mins)"},
            number = {'font': {'color': "white" if health['Data_Lag'] < 15 else "red"}}
        ), row=3, col=3)

        # LAYOUT STYLING (The "Premium" Look)
        fig.update_layout(
            template="plotly_dark",
            height=1200,
            title_text=f"CHIMERA PRIME: COMMAND CENTER | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            font=dict(family="Roboto Mono, monospace"),
            paper_bgcolor="#050505",
            plot_bgcolor="#0a0a0a",
            margin=dict(l=30, r=30, t=80, b=30)
        )
        
        output_file = "chimera_prime_dashboard.html"
        fig.write_html(output_file)
        print(f"COMMAND CENTER ONLINE: {output_file}")

# --- LAUNCH ---
if __name__ == "__main__":
    dashboard = ChimeraCommandCenter()
    dashboard.generate_dashboard()
