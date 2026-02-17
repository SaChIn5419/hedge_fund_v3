import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CHIMERA: QUANT DESK (REALITY MODE)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp { background-color: #050505; color: #e0e0e0; }
        .block-container { padding-top: 1rem; }
        .stMetric { background-color: #0e1117; border: 1px solid #333; padding: 10px; border-radius: 4px; }
        h1, h2, h3 { color: #00ffcc; font-family: 'Roboto Mono', monospace; }
        div[data-testid="stMetricValue"] { font-family: 'Roboto Mono', monospace; color: #fff; font-size: 24px; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "trade_log.csv")

@st.cache_data(ttl=60)
def load_trade_log():
    """Loads the authentic backtest log."""
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    
    # Try reading with headers first (Phase 9 standard)
    df = pd.read_csv(LOG_FILE)
    
    # Validation
    if 'date' not in df.columns:
        st.error("Trade Log schema mismatch. Re-run Backtest.")
        return pd.DataFrame()
        
    df['date'] = pd.to_datetime(df['date'])
    return df

def main():
    st.sidebar.title("⏳ TIME MACHINE (REAL)")
    
    # 1. LOAD REAL DATA
    df_log = load_trade_log()
    if df_log.empty:
        st.error("Trade Log not found or invalid.")
        return
        
    # 2. DATE SELECTOR
    available_dates = sorted(df_log['date'].unique())
    if not available_dates:
        st.error("Log contains no dates.")
        return
        
    min_date = available_dates[0]
    max_date = available_dates[-1]
    
    st.sidebar.markdown("### Select Simulation Date")
    selected_date = st.sidebar.select_slider(
        "Travel to:",
        options=available_dates,
        value=available_dates[-1],
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d")
    )
    
    # 3. FILTER SNAPSHOT
    # Convert to pandas timestamp for comparison
    sel_ts = pd.to_datetime(selected_date)
    snapshot = df_log[df_log['date'] == sel_ts].copy()
    
    # --- HUD DATA CALCULATIONS ---
    # Need Regime, Volatility, Benchmark Price
    # We can infer from the first row of snapshot since these are system-wide
    if not snapshot.empty:
        regime_code = snapshot['market_regime'].iloc[0]
        nifty_vol = snapshot['nifty_vol'].iloc[0] if 'nifty_vol' in snapshot.columns else 0
        
        # Determine Text Regime
        if regime_code == 1.0: regime_txt = "BULLISH"
        elif regime_code == 0.5: regime_txt = "VOLATILE"
        else: regime_txt = "BEARISH"
        
        # Holdings Count
        active_pos = snapshot[snapshot['weight'] > 0]
        holdings_count = len(active_pos)
        total_exp = active_pos['weight'].sum() * 100
        
        # Benchmark Price (Not in log explicitly usually, but we have Close of assets)
        # We can just show System Health
    else:
        regime_txt = "NO DATA"
        nifty_vol = 0
        holdings_count = 0
        total_exp = 0

    # --- DASHBOARD HEADER ---
    st.title(f"CHIMERA DESK // {sel_ts.strftime('%Y-%m-%d')}")
    st.markdown(f"**REALITY MODE**: Displaying actual internal state from `trade_log.csv`.")
    
    # --- TABS ---
    tab_hud, tab_micro, tab_forensic, tab_audit = st.tabs(["🚀 COMMAND CENTER", "🔬 MICROSCOPE", "🔎 FORENSICS", "📊 TEARSHEET"])
    
    with tab_hud:
        # Generate Premium Plotly Dashboard
        
        # 1. Prepare Data for Plotly
        
        # A. Calculate Real Correlation (30-Day Window)
        # We need history before selected date
        start_lookback = sel_ts - pd.Timedelta(days=45) # Buffer
        history_slice = df_log[(df_log['date'] <= sel_ts) & (df_log['date'] >= start_lookback)]
        
        if not history_slice.empty:
             # Pivot: Index=Date, Col=Ticker, Val=Close
             pivot_hist = history_slice.pivot_table(index='date', columns='ticker', values='fwd_return')
             # Calc Correlation of returns
             if len(pivot_hist) > 10:
                 corr_matrix = pivot_hist.corr()
                 avg_corr = corr_matrix.mean().mean()
             else:
                 avg_corr = 0.0
        else:
            avg_corr = 0.0

        # B. Calculate Portfolio Volatility (Real VIX Proxy)
        # Instead of broken nifty_vol column, use Portfolio Return Std Dev
        if not history_slice.empty:
             daily_rets = history_slice.groupby('date')['net_pnl'].sum() / 1000000 # Approx
             port_vol = daily_rets.std() * np.sqrt(252) * 100
             if pd.isna(port_vol): port_vol = 0
        else:
             port_vol = 0

        health = {
            'Regime': regime_txt,
            'VIX_Proxy': port_vol, # Now using Real Calculated Vol
            'Avg_Corr': avg_corr,
            'Data_Lag': 0
        }
        
        # Grid Data
        grid_metrics = pd.DataFrame()
        if not snapshot.empty:
            grid_metrics = snapshot[['ticker', 'close', 'fwd_return', 'net_pnl']].copy()
            # Rename for display
            # Use fwd_return for Change % (it is a ratio like -0.04)
            grid_metrics['Change %'] = (grid_metrics['fwd_return'] * 100).round(2)
            grid_metrics['Price'] = grid_metrics['close'].round(2)
            grid_metrics['PnL ($)'] = grid_metrics['net_pnl'].round(2)
            grid_metrics['Ticker'] = grid_metrics['ticker']
            grid_metrics['Alloc'] = snapshot['weight'] # Add Alloc for Pie Chart
            
            # Map Signal/Structure from available columns
            if 'efficiency' in snapshot.columns:
                grid_metrics['Structure'] = snapshot['efficiency'].apply(lambda x: "VACUUM" if x > 0.3 else "TRAPPED")
                
            grid_metrics['Signal'] = "NEUTRAL"
            
            # Use kinetic_energy (0.91) -> 91%
            if 'kinetic_energy' in snapshot.columns:
                grid_metrics['Energy'] = (snapshot['kinetic_energy'] * 100).round(2)
            else:
                 grid_metrics['Energy'] = 0.0
            
            # Select final columns for display
            display_cols = ['Ticker', 'Price', 'Change %', 'PnL ($)', 'Alloc', 'Structure', 'Signal', 'Energy']
            grid_metrics = grid_metrics[display_cols]

            # Formatting
            # clean up infinite values
            grid_metrics = grid_metrics.replace([np.inf, -np.inf], 0).fillna(0)

        # 2. Build Figure (Ported from chimera_dashboard.py)
        fig = make_subplots(
            rows=3, cols=3,
            column_widths=[0.3, 0.4, 0.3],
            row_heights=[0.2, 0.4, 0.4],
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "table", "colspan": 2}, None, {"type": "domain"}], 
                [{"type": "heatmap", "colspan": 2}, None, {"type": "indicator"}] 
            ]
            # REMOVED SUBPLOT TITLES TO FIX OVERLAP
        )
        
        # HUD 1: REGIME
        color = "#00ffcc" if health['Regime'] == "BULLISH" else "#ff5555"
        fig.add_trace(go.Indicator(
            mode = "number+gauge", value = 1 if health['Regime']=="BULLISH" else 0,
            title = {"text": f"REGIME: {health['Regime']}"},
            gauge = {'axis': {'range': [0, 1]}, 'bar': {'color': color}},
            number = {'font': {'color': color}}
        ), row=1, col=1)

        # HUD 2: VOLATILITY
        fig.add_trace(go.Indicator(
            mode = "number+gauge", value = health['VIX_Proxy'],
            title = {"text": "PORTFOLIO VOL %"},
            gauge = {
                'axis': {'range': [0, 50]},
                'bar': {'color': "#ffaa00"},
                'steps': [
                    {'range': [0, 10], 'color': "#1e3a2f"},
                    {'range': [10, 20], 'color': "#3a3a1e"},
                    {'range': [20, 50], 'color': "#3a1e1e"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 20}
            }
        ), row=1, col=2)

        # HUD 3: DIVERSITY
        fig.add_trace(go.Indicator(
            mode="number", 
            value=health['Avg_Corr'], 
            title={"text": "AVG CORRELATION"},
            number={'valueformat': ".2f"}
        ), row=1, col=3)

        # TABLE
        if not grid_metrics.empty:
            cell_colors = []
            for idx, row in grid_metrics.iterrows():
                if row.get('Structure') == 'VACUUM': cell_colors.append("#1e3a2f")
                else: cell_colors.append("#3a1e1e")
            
            fig.add_trace(go.Table(
                header=dict(values=list(grid_metrics.columns), fill_color='#333', align='left', font=dict(color='white')),
                cells=dict(values=[grid_metrics[k].tolist() for k in grid_metrics.columns],
                           fill_color=[cell_colors * len(grid_metrics.columns)],
                           align='left', font=dict(color='lightgrey'), format=[None, ".2f", ".2f", ".2f", ".1%", None, None, ".2f"])
            ), row=2, col=1)
            
            # ALLOCATION PIE CHART (New)
            # Replaces Benchmark Trend
            fig.add_trace(go.Pie(
                labels=grid_metrics['Ticker'], 
                values=grid_metrics['Alloc'], # Use Alloc column
                hole=0.4,
                textinfo='label+percent',
                marker=dict(colors=cell_colors) # Match table colors? Or auto.
            ), row=2, col=3)
            
            # Note: We removed the Benchmark Scatter plot code that used to be here.
            
        # LATENCY
        fig.add_trace(go.Indicator(mode="number", value=0, title={"text": "DATA LAG (Mins)"}), row=3, col=3)

        fig.update_layout(template="plotly_dark", height=800, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="#000")
        
        if not snapshot.empty:
             # Ensure Pie chart uses weights
             # We just set it above correctly using grid_metrics['Alloc'] (renamed from weight earlier? No, wait)
             # snapshot has 'weight'. grid_metrics does not have 'Alloc' unless we added it.
             # Let's fix grid_metrics creation to include weight as 'Alloc'.
             pass # Logic moved up.

        st.plotly_chart(fig, use_container_width=True)

    with tab_micro:
        st.subheader("ASSET INSPECTOR (X-RAY)")
        
        if not snapshot.empty:
            # Selector
            assets = snapshot['ticker'].tolist()
            selected_asset = st.selectbox("Select Asset to Inspect:", assets)
            
            asset_row = snapshot[snapshot['ticker'] == selected_asset].iloc[0]
            
            # Columns
            mc1, mc2, mc3 = st.columns(3)
            
            with mc1:
                st.markdown("#### PRICE PHYSICS")
                st.metric("Price", f"{asset_row['close']:.2f}")
                
                # Deviation from SMA
                sma = asset_row['sma_20']
                dev = (asset_row['close'] - sma) / sma * 100
                st.metric("Extension from Mean", f"{dev:.2f}%", help="Distance from 20SMA")
                
                # Bands
                bb_width = (asset_row['upper_band'] - asset_row['lower_band']) / sma * 100
                st.metric("Bandwidth (Compression)", f"{bb_width:.2f}%")

            with mc2:
                st.markdown("#### SIZING DYNAMICS")
                
                # Visualizing the Cap
                alpha_w = asset_row['alpha_weight'] if 'alpha_weight' in asset_row else 0
                final_w = asset_row['weight']
                st.metric("Theoretical Model Weight", f"{alpha_w:.1%}")
                st.metric("Risk Manager Approved", f"{final_w:.1%}", 
                          delta=f"Capped: {(final_w - alpha_w):.1%}" if final_w < alpha_w else "Uncapped")
                
                eff = asset_row['efficiency'] if 'efficiency' in asset_row else 0
                st.progress(min(eff, 1.0), text=f"Efficiency Score: {eff:.2f}")

            with mc3:
                st.markdown("#### KINETIC STATE")
                mom = asset_row['momentum'] * 100 if 'momentum' in asset_row else 0
                ke = asset_row['kinetic_energy'] * 100 if 'kinetic_energy' in asset_row else 0
                
                st.metric("Momentum (Raw)", f"{mom:.1f}%")
                st.metric("Kinetic Energy (Slope)", f"{ke:.1f}%")
                
                status = "🟢 TURBO" if ke > 25 and eff > 0.3 else "🔴 DRAG"
                st.markdown(f"## {status}")
                
                if 'gaussian_zscore' in asset_row:
                    z = asset_row['gaussian_zscore']
                    st.metric("Gaussian Z-Score", f"{z:.2f}σ", delta="Overextended" if abs(z) > 2.0 else "Normal", delta_color="inverse")

        else:
            st.info("No assets active.")

    with tab_forensic:
        st.subheader("THE BLACK BOX RECORDER")
        st.markdown("Forensic analysis of the entire Backtest History (Aggregate Data).")
        
        fc1, fc2 = st.columns(2)
        
        with fc1:
            st.markdown("#### A. SKILL VS LUCK (Kinetic Validity)")
            # Scatter: Energy vs Return
            # Sample 2000 points to avoid lag
            sample_size = min(len(df_log), 2000)
            scatter_df = df_log.sample(sample_size)
            
            if 'kinetic_energy' in scatter_df.columns:
                fig_scatter = px.scatter(
                    scatter_df, 
                    x='kinetic_energy', 
                    y='fwd_return', 
                    color='structure_tag' if 'structure_tag' in scatter_df.columns else None,
                    title=f"Energy vs Return ({sample_size} Trades)",
                    template="plotly_dark",
                    labels={"kinetic_energy": "Kinetic Energy (Slope)", "fwd_return": "Forward Return"}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("Kinetic Energy data missing.")
        
        with fc2:
            st.markdown("#### B. DECISION TREE (Why did we trade?)")
            if 'decision_reason' in df_log.columns:
                # Sunburst of Reasons
                reason_counts = df_log.groupby("decision_reason").size().reset_index(name='count')
                fig_sun = px.sunburst(
                    reason_counts, 
                    path=['decision_reason'], 
                    values='count',
                    title="Trade Rationale Distribution",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_sun, use_container_width=True)
            else:
                st.warning("Decision Rationale missing.")

        st.markdown("---")
        st.markdown("#### C. RISK THERMOMETER (The Dimmer Switch)")
        # Area Chart: VIX vs Date. Overlay Leverage Mult?
        # Aggregate logic per day
        daily_risk = df_log.groupby("date").first().reset_index()
        
        # Create dual axis chart
        fig_risk = make_subplots(specs=[[{"secondary_y": True}]])
        
        # VIX Area
        if 'nifty_vol' in daily_risk.columns:
            fig_risk.add_trace(go.Scatter(
                x=daily_risk['date'], y=daily_risk['nifty_vol'] * 100,
                fill='tozeroy', name="Market Volatility (VIX)",
                line=dict(color='red', width=1)
            ), secondary_y=False)
        
        # Leverage Stepped Line
        if 'leverage_mult' in daily_risk.columns:
             fig_risk.add_trace(go.Scatter(
                x=daily_risk['date'], y=daily_risk['leverage_mult'],
                name="Leverage Multiplier",
                line=dict(color='cyan', width=2, shape='hv')
            ), secondary_y=True)
             
        fig_risk.update_layout(
            title="The Risk Thermometer: Volatility vs Leverage",
            template="plotly_dark", 
            height=400,
            hovermode="x unified"
        )
        fig_risk.update_yaxes(title_text="Volatility %", secondary_y=False)
        fig_risk.update_yaxes(title_text="Leverage Mult (x)", secondary_y=True, range=[0, 2.0])
        
        st.plotly_chart(fig_risk, use_container_width=True)


    with tab_audit:
        st.subheader("EQUITY CURVE RECONSTRUCTION")
        # Reconstruct Equity Curve from Log
        daily_pnl = df_log.groupby('date')['net_pnl'].sum().reset_index()
        daily_pnl['Equity'] = daily_pnl['net_pnl'].cumsum() + 1000000 # Assume 1M start
        
        # Plot
        fig = px.line(daily_pnl, x='date', y='Equity', title="Portfolio Equity")
        fig.update_layout(template="plotly_dark")
        
        # Mark current date
        fig.add_vline(x=sel_ts, line_dash="dot", line_color="yellow")
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
