import streamlit as st
import pandas as pd
import sys
import os

# Add parent dir to path to import chimera_live
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chimera_live import ChimeraLiveProtocol

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
    with st.expander("🔍 VIEW RAW TELEMETRY"):
        st.dataframe(df[df['date'] == selected_date])

if __name__ == "__main__":
    main()
