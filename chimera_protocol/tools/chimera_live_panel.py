import streamlit as st
import pandas as pd
import sys
import os
import asyncio
import time
from datetime import datetime

# Add parent dir to path to import chimera_live
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from chimera_live import ChimeraLiveProtocol
from chimera_execution.broker_angleone_async import AsyncAngleOneBroker
from chimera_protocol.chimera_engine import validate_vacuum_with_depth
try:
    from Chimera_Pro_Live.chimera_governance.capital_brain import CapitalAllocationBrain
except ImportError:
    # Fallback mock if path issue
    class CapitalAllocationBrain:
        def __init__(self): pass
        def update_allocation(self, data): return 1.0
try:
    from chimera_protocol.data_foundry import ChimeraFoundry
    FOUNDRY = ChimeraFoundry()
except:
    FOUNDRY = None

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CHIMERA: LIVE CONTROL",
    page_icon="☢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp { background-color: #000000; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
        
        /* STATUS INDICATORS */
        .status-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            border: 2px solid #333;
        }
        .status-green { background-color: #004d40; color: #00e676; border-color: #00e676; }
        .status-yellow { background-color: #4a4a00; color: #ffea00; border-color: #ffea00; }
        .status-red { background-color: #4d0000; color: #ff1744; border-color: #ff1744; }

        /* LAYER CARDS */
        .layer-card {
            background-color: #111;
            border: 1px solid #333;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .layer-title {
            color: #888;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            border-bottom: 1px solid #222;
        }
        
        /* ALERTS */
        .alert-item {
            padding: 10px;
            margin-bottom: 5px;
            background: #1a1a1a;
            border-left: 4px solid #555;
            font-size: 14px;
        }
        .alert-red { border-left-color: #ff1744; }
        .alert-yellow { border-left-color: #ffea00; }

        /* ACTION BOX */
        .action-box {
            background-color: #0d47a1;
            color: #fff;
            padding: 20px;
            border-radius: 5px;
            font-size: 18px;
            border: 1px solid #2979ff;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chimera_blackbox_final.csv")

@st.cache_data
def load_data():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    return pd.read_csv(LOG_FILE)

# --- ANGEL ONE BRIDGE ---
@st.cache_resource
@st.cache_resource
def get_broker():
    """
    Returns a cached, authenticated broker instance.
    """
    broker = AsyncAngleOneBroker()
    if broker.login():
        return broker
    return None

def main():
    st.title("☢️ CHIMERA LIVE CONTROL PANEL")
    
    df = load_data()
    if df.empty:
        st.error("NO TELEMETRY DATA FOUND. SYSTEM OFFLINE.")
        return

    # Date Selector for Historical Replay
    df['date'] = pd.to_datetime(df['date'])
    dates = sorted(df['date'].unique())
    selected_date = st.sidebar.select_slider(
        "SIMULATION DATE",
        options=dates,
        value=dates[-1],
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d")
    )
    
    # LIVE MODE TOGGLE
    live_mode = st.sidebar.checkbox("🔴 GO LIVE (ANGEL ONE)", value=False)
    sim_mode = st.sidebar.checkbox("🧪 SIMULATE DEPTH (OFFLINE TEST)", value=False)
    
    if live_mode and sim_mode:
        st.sidebar.error("Select only ONE mode.")
        live_mode = False
        sim_mode = False
    
    broker = None
    if live_mode:
        st.sidebar.warning("⚠️ CONNECTING TO LIVE MARKETS...")
        broker = get_broker()
        if broker:
            st.sidebar.success("✅ CONNECTED")
        else:
            st.sidebar.error("❌ CONNECTION FAILED")
            live_mode = False

    # --- EXECUTE PROTOCOL ---
    protocol = ChimeraLiveProtocol(df)
    protocol.run_protocol_check(current_date=selected_date)
    report = protocol.get_report()
    
    # --- DISPLAY STATUS ---
    status = report['status']
    
    if status == "GREEN":
        st.markdown(f'<div class="status-box status-green">SYSTEM NORMAL<br><span style="font-size:16px">FULL DEPLOYMENT AUTHORIZED</span></div>', unsafe_allow_html=True)
    elif status == "YELLOW":
        st.markdown(f'<div class="status-box status-yellow">CAUTIONARY REGIME<br><span style="font-size:16px">REDUCE LEVERAGE / DEFENSIVE POSTURE</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-box status-red">CRITICAL PROTOCOL FAILURE<br><span style="font-size:16px">IMMEDIATE INTERVENTION REQUIRED</span></div>', unsafe_allow_html=True)

    # --- ACTIONS REQUIRED ---
    if report['actions']:
        st.error("### ⚠️ REQUIRED ACTIONS")
        for action in report['actions']:
            st.markdown(f"- **{action}**")
    else:
        st.success("### ✅ NO ACTIONS REQUIRED")

    st.markdown("---")

    # --- GOVERNANCE LAYER (INTEGRATED) ---
    st.markdown("### 🧠 CAPITAL BRAIN AI")
    c_brain_1, c_brain_2 = st.columns([1, 3])
    
    with c_brain_1:
         # MAP STATUS TO REGIME
         regime_map = {
             "GREEN": "TRENDING",
             "YELLOW": "STABLE", # Cautionary
             "RED": "CHAOS"
         }
         current_regime = regime_map.get(status, "STABLE")
         
         # Instantiate Brain
         brain = CapitalAllocationBrain()
         
         # Get Recommendation
         # We need volatility. Let's use a proxy or default 0.015 (1.5%) if not in df
         # If Nifty Vol available, use it.
         vol_proxy = 0.015
         if 'nifty_vol' in df.columns:
             try:
                 vol_proxy = df['nifty_vol'].iloc[-1]
             except: pass
             
         rec_leverage = brain.update_allocation({"regime": current_regime, "volatility": vol_proxy})
         
         st.metric("AI RECOMMENDED LEVERAGE", f"{rec_leverage:.1f}x", delta=f"Regime: {current_regime}")

    with c_brain_2:
        st.info(f"**LOGIC:** The Capital Brain has analyzed the market state as **{current_regime}**. Based on Volatility ({vol_proxy:.1%}), it recommends resizing all positions to **{rec_leverage}x** base sizing.")

    st.markdown("---")

    # --- LAYER BREAKDOWN ---
    c1, c2, c3 = st.columns(3)
    
    # LAYER 1
    with c1:
        l1_status = report['layer_status']['L1_Signal']
        color = "#00e676" if l1_status == "GREEN" else ("#ffea00" if l1_status == "YELLOW" else "#ff1744")
        st.markdown(f'<div class="layer-card"><div class="layer-title" style="color:{color}">LAYER 1: SIGNAL PHYSICS</div>', unsafe_allow_html=True)
        st.caption("Energy Integrity | Efficiency Decay | Structure Vacuum")
        
        # Alerts for L1
        alerts = [a for a in report['alerts'] if a['layer'] == 'Layer 1']
        if not alerts:
            st.markdown("✅ ALL SUBSYSTEMS NOMINAL")
        else:
            for a in alerts:
                c = "alert-red" if a['status'] == "RED" else "alert-yellow"
                st.markdown(f'<div class="alert-item {c}"><b>{a["rule"]}</b><br>{a["message"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # LAYER 2
    with c2:
        l2_status = report['layer_status']['L2_Execution']
        color = "#00e676" if l2_status == "GREEN" else ("#ffea00" if l2_status == "YELLOW" else "#ff1744")
        st.markdown(f'<div class="layer-card"><div class="layer-title" style="color:{color}">LAYER 2: EXECUTION REALITY</div>', unsafe_allow_html=True)
        st.caption("Slippage | Turnover Shock | Leverage Clamp")
        
        alerts = [a for a in report['alerts'] if a['layer'] == 'Layer 2']
        if not alerts:
            st.markdown("✅ EXECUTION PARAMETERS NORMAL")
        else:
            for a in alerts:
                c = "alert-red" if a['status'] == "RED" else "alert-yellow"
                st.markdown(f'<div class="alert-item {c}"><b>{a["rule"]}</b><br>{a["message"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # LAYER 3
    with c3:
        l3_status = report['layer_status']['L3_Survival']
        color = "#00e676" if l3_status == "GREEN" else ("#ffea00" if l3_status == "YELLOW" else "#ff1744")
        st.markdown(f'<div class="layer-card"><div class="layer-title" style="color:{color}">LAYER 3: CAPITAL SURVIVAL</div>', unsafe_allow_html=True)
        st.caption("Rolling Sharpe | Drawdown Ladder | Regime Flip")
        
        alerts = [a for a in report['alerts'] if a['layer'] == 'Layer 3']
        if not alerts:
            st.markdown("✅ SAFETY PROTOCOLS ENGAGED")
        else:
            for a in alerts:
                c = "alert-red" if a['status'] == "RED" else "alert-yellow"
                st.markdown(f'<div class="alert-item {c}"><b>{a["rule"]}</b><br>{a["message"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # --- LIVE MICROSTRUCTURE VIEW ---
    current_positions = df[df['date'] == selected_date]
    
    with st.expander("🔍 VIEW TELEMETRY & MICROSTRUCTURE", expanded=True):
        if not live_mode and not sim_mode:
            st.dataframe(current_positions)
        else:
            if sim_mode:
                st.info("🧪 RUNNING SIMULATION: GENERATING MOCK MICROSTRUCTURE DATA")
            else:
                st.info("📡 FETCHING LIVE MARKET DEPTH (ANGEL ONE)...")
            
            # Prepare container for live data
            live_data_list = []
            
            # Create a progress bar
            prog_bar = st.progress(0)
            
            # Filter for active positions only
            active_pos = current_positions[current_positions['ticker'] != 'CASH']
            total = len(active_pos)
            
            broker = get_broker() if live_mode else None
            
            import random # For sim
            
            for idx, row in enumerate(active_pos.itertuples()):
                ticker = row.ticker
                micro = None
                
                if live_mode:
                    # Fetch Microstructure
                    try:
                        micro = asyncio.run(broker.get_market_microstructure("NSE", ticker))
                    except Exception as e:
                        print(f"Error fetching {ticker}: {e}")
                elif sim_mode:
                    # Simulate Data
                    # Randomize friction to show both Safe and Blocked scenarios
                    rand_friction = random.uniform(0.5, 5.0) 
                    micro = {
                        'ltp': row.close * random.uniform(0.99, 1.01),
                        'friction': rand_friction,
                        'total_buy_pressure': 100000,
                        'total_sell_pressure': 100000 * rand_friction
                    }
                
                item = {
                    'Ticker': ticker,
                    'Weight': row.weight,
                    'Log_Close': row.close,
                    'Signal': 'BULL_TURBO', # Assuming active means bull/turbo
                    'Structure': row.structure_tag
                }
                
                if micro:
                    # RECORD TO DATA LAKE
                    if live_mode and FOUNDRY:
                         # timestamp logic if not present
                         if 'timestamp' not in micro:
                             micro['timestamp'] = datetime.now()
                         FOUNDRY.record_depth_snapshot(ticker, micro)

                    item['Live_LTP'] = micro['ltp']
                    item['Friction'] = f"{micro['friction']:.2f}"
                    # item['Buy_Qty'] = micro['total_buy_pressure']
                    # item['Sell_Qty'] = micro['total_sell_pressure']
                    
                    # PnL Calc
                    entry = row.close 
                    curr = micro['ltp']
                    ret = (curr - entry) / entry
                    item['Live_PnL%'] = f"{ret*100:.2f}%"
                    
                    # VALIDATE VACUUM
                    is_safe = validate_vacuum_with_depth(item['Signal'], micro)
                    if is_safe:
                        item['Vacuum_Status'] = "✅ REAL"
                    else:
                        item['Vacuum_Status'] = "⛔ FAKE (WALL DETECTED)"
                    
                else:
                    item['Live_LTP'] = "Offline"
                    item['Friction'] = "N/A"
                    item['Vacuum_Status'] = "❓ UNKNOWN"

                live_data_list.append(item)
                prog_bar.progress((min(idx + 1, total)) / total)
            
            live_df = pd.DataFrame(live_data_list)
            st.dataframe(live_df)
            
            # Friction Warning
            if 'Friction' in live_df.columns:
                 # Clean column for check
                 live_df['Friction_Val'] = pd.to_numeric(live_df['Friction'], errors='coerce')
                 high_fri = live_df[live_df['Friction_Val'] > 3.0]
                 if not high_fri.empty:
                     st.error(f"⚠️ HIGH FRICTION (WALLS) DETECTED IN: {', '.join(high_fri['Ticker'].tolist())}")
                     st.caption("These trades would be BLOCKED by the Execution Engine.")

if __name__ == "__main__":
    main()
