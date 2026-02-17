import streamlit as st
import json
import pandas as pd
import time
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="Chimera Cortex")

# Custom CSS for "Dark/Cyberpunk" feel
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .stMetric {
        background-color: #1e2127;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #30333d;
    }
</style>
""", unsafe_allow_html=True)

# Path to Telemetry
NEURAL_STATE_PATH = "data/neural_state.json"

def load_state():
    try:
        with open(NEURAL_STATE_PATH, "r") as f:
            return json.load(f)
    except:
        return None

# Title
st.title("🧠 Chimera Institutional Cortex")

# Auto-Refresh Placeholder
placeholder = st.empty()

while True:
    state = load_state()
    
    with placeholder.container():
        if not state:
            st.error("Waiting for Neural State...")
            time.sleep(1)
            continue
            
        # --- HEADER ---
        ts = state.get('timestamp', 'N/A')
        latency = "OK" # Placeholder calculation
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("System Heartbeat", ts.split('.')[0])
        c2.metric("Market Depth", state.get('market_depth', 0))
        
        # --- GOVERNANCE LAYER ---
        st.markdown("---")
        st.subheader("🏛️ Governance & Risk")
        
        regime = state.get('regime', {})
        regime_name = regime.get('regime', 'INIT') if isinstance(regime, dict) else regime
        entropy = regime.get('entropy', 0.0) if isinstance(regime, dict) else 0.0
        leverage = state.get('leverage', 1.0)
        
        g1, g2, g3 = st.columns(3)
        
        # Color Logic
        regime_color = "normal"
        if regime_name == "CHAOS": regime_color = "inverse"
        if regime_name == "TRENDING": regime_color = "off" # Streamlit doesn't have specific colors, just logic
        
        g1.metric("Market Regime", regime_name, delta=f"Entropy: {entropy:.2f}", delta_color=regime_color)
        g2.metric("Target Leverage", f"{leverage}x")
        
        # --- EXECUTION LAYER ---
        st.markdown("---")
        st.subheader("⚙️ Execution Engine")
        # Execution State
        exec_state = state.get('execution_state', {})
        if isinstance(exec_state, str):
            current_state = exec_state
            active_orders = 0 # If exec_state is a string, there's no 'active_orders' key
        else:
            current_state = exec_state.get('state', 'UNKNOWN')
            active_orders = exec_state.get('active_orders', 0)
            
        st.info(f"System State: **{current_state}**")
        
        e1, e2 = st.columns(2)
        e1.metric("State Machine", current_state)
        e2.metric("Active Orders", active_orders)
        
        # --- PHYSICS LAYER ---
        st.markdown("---")
        st.subheader("⚛️ Physics Signal")
        
        last_physics = state.get('last_physics', 'No Signal')
        
        if isinstance(last_physics, dict):
            p1, p2, p3 = st.columns(3)
            p1.metric("Token", last_physics.get('token', 'N/A'))
            p1.metric("Energy", f"{last_physics.get('energy', 0):.2f}")
            p2.metric("Structure", last_physics.get('structure', 'N/A'))
            p3.metric("Signal", last_physics.get('signal', 'N/A'))
        else:
            st.info("Waiting for Candle Closure...")

    time.sleep(1)
