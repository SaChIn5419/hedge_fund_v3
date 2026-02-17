# reality_control_flow.py
# Chimera Reality Control Flow - Streamlit app
# Shows per-layer leave-one-out influence on final scale over time

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import datetime

LOG_DIR = Path("logs")
NEURAL_FILE = LOG_DIR / "neural_state.json"
STABILITY_FILE = LOG_DIR / "stability_log.csv"
OUT_CSV = LOG_DIR / "reality_control_flow_output.csv"

st.set_page_config(layout="wide", page_title="Chimera Reality Control Flow")
st.title("🔬 Chimera — Reality Control Flow")

# ---------- helper functions ----------
def load_neural_state():
    """Load neural_state.json. Accepts either single JSON object (latest)
       or newline-delimited JSON or an array of objects (time-series)."""
    if not NEURAL_FILE.exists():
        return None
    try:
        raw = NEURAL_FILE.read_text()
        parsed = json.loads(raw)
        # If parsed is dict with 'timestamp' or time series as list, normalize
        if isinstance(parsed, dict):
            # if contains a time series keyed by timestamp, try to convert
            # if it's a single snapshot, wrap in a list and require a 'timestamp'
            if 'timestamp' in parsed or 'time' in parsed:
                return pd.DataFrame([parsed])
            else:
                # try to detect if it's a mapping of timestamps -> states
                # else treat as single snapshot
                # fallback: wrap into one-row DataFrame
                return pd.DataFrame([parsed])
        elif isinstance(parsed, list):
            return pd.DataFrame(parsed)
        else:
            return None
    except Exception:
        # maybe newline-delimited JSON (NDJSON)
        try:
            rows = []
            for line in NEURAL_FILE.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
            return pd.DataFrame(rows) if rows else None
        except Exception:
            return None

def load_stability_csv():
    if not STABILITY_FILE.exists():
        return None
    try:
        df = pd.read_csv(STABILITY_FILE)
        return df
    except Exception:
        return None

def pick_time_series():
    # priority: neural_state.json (with per-row scales) -> stability_log.csv -> None
    ns = load_neural_state()
    if ns is not None and len(ns) > 0:
        st.sidebar.success("Using logs/neural_state.json as source")
        return ns
    st.sidebar.info("neural_state.json not found or empty, trying stability_log.csv")
    stbl = load_stability_csv()
    if stbl is not None and len(stbl) > 0:
        st.sidebar.success("Using logs/stability_log.csv as source")
        return stbl
    st.sidebar.error("No suitable logs found. Please provide logs/neural_state.json or logs/stability_log.csv")
    return None

def ensure_columns(df, expected_cols):
    # ensure expected column names exist (lowercasing test)
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for e in expected_cols:
        if e in cols:
            mapping[e] = cols[e]
        else:
            # fuzzy matches
            candidates = [c for c in df.columns if e in c.lower()]
            mapping[e] = candidates[0] if candidates else None
    return mapping

def compute_final_scale_from_row(row):
    # If row already contains 'final_scale', use it
    if 'final_scale' in row and not pd.isna(row['final_scale']):
        return float(row['final_scale'])
    # else try to reconstruct via consciousness arbitration proxies:
    # many logs already include precomputed final. If not, we approximate as geometric mean of scales.
    parts = []
    for k in ['genome_scale','drift_scale','capital_scale','adaptive_scale','meta_scale']:
        if k in row and not pd.isna(row[k]):
            try:
                parts.append(float(row[k]))
            except:
                pass
    if parts:
        # use multiplicative model default as proxy
        return float(np.prod(parts) ** (1.0/len(parts)))
    return 1.0

def neutralize_scale(row, key):
    """Return a copy of row with the given key neutralized (set to 1.0)."""
    r = row.copy()
    r[key] = 1.0
    return r

def compute_influence_timeseries(df, layer_keys):
    """Given a dataframe with per-timestamp layer scales and final_scale,
       compute leave-one-out influence for each layer.
       Influence_i = final_scale(actual) - final_scale(with layer_i neutralized)
    """
    out_rows = []
    for idx, r in df.iterrows():
        base_final = compute_final_scale_from_row(r)
        influences = {}
        for k in layer_keys:
            if k not in r or pd.isna(r[k]):
                influences[k] = 0.0
                continue
            neutral = neutralize_scale(r, k)
            neutral_final = compute_final_scale_from_row(neutral)
            # influence is the change in final scale attributable to layer k
            # positive means that layer increased the final scale; negative means it decreased it
            influences[k] = float(base_final - neutral_final)
        row_out = {'index': idx, 'timestamp': r.get('timestamp', r.get('time', None)), 'base_final': base_final}
        row_out.update(influences)
        out_rows.append(row_out)
    res = pd.DataFrame(out_rows)
    # convert timestamp if needed
    if 'timestamp' in res.columns:
        try:
            res['timestamp'] = pd.to_datetime(res['timestamp'], unit='s', errors='coerce')
        except:
            res['timestamp'] = pd.to_datetime(res['timestamp'], errors='coerce')
    return res

# ---------- main ----------
df = pick_time_series()
if df is None:
    st.stop()

# normalize column names to lower-case for known keys
df.columns = [c.lower() for c in df.columns]

expected = ['genome_scale','drift_scale','capital_scale','adaptive_scale','meta_scale','final_scale']
mapping = ensure_columns(df, expected)

# rename found columns to canonical names for simplicity
renames = {}
for k, v in mapping.items():
    if v and v != k:
        renames[v] = k
df = df.rename(columns=renames)

# fill missing numeric columns with NaN
for k in expected:
    if k not in df.columns:
        df[k] = np.nan

# If no explicit timestamp, create one from index
if 'timestamp' not in df.columns:
    df['timestamp'] = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=len(df)), periods=len(df))

# compute influences
layer_keys = [k for k in expected if k.endswith('_scale') and k != 'final_scale']
influence_ts = compute_influence_timeseries(df, layer_keys)

# For stability & stacked area plotting we want the signed influences normalized to absolute sum per row
# but keep raw signed influences as well.
influence_ts = influence_ts.set_index('timestamp').sort_index()
signed = influence_ts[layer_keys].fillna(0)

# compute total absolute influence per row to normalize (avoid divide by zero)
abs_sum = signed.abs().sum(axis=1).replace({0:1.0})
normed = signed.div(abs_sum, axis=0)

# Compose combined DataFrame for plotting
plot_df = normed.reset_index().melt(id_vars='timestamp', var_name='layer', value_name='normed_influence')
raw_df = signed.reset_index().melt(id_vars='timestamp', var_name='layer', value_name='signed_influence')

# Add small smoothing option
smooth_window = st.sidebar.slider("Smoothing window (rows)", 1, 51, 5, 2)
if smooth_window > 1:
    tmp = plot_df.pivot(index='timestamp', columns='layer', values='normed_influence').rolling(smooth_window, min_periods=1).mean()
    plot_df = tmp.reset_index().melt(id_vars='timestamp', var_name='layer', value_name='normed_influence')
    tmp2 = raw_df.pivot(index='timestamp', columns='layer', values='signed_influence').rolling(smooth_window, min_periods=1).mean()
    raw_df = tmp2.reset_index().melt(id_vars='timestamp', var_name='layer', value_name='signed_influence')

# ---------- UI: stacked area ----------
st.markdown("## 1) Stacked Area — Normalized Influence Over Time")
fig_area = px.area(plot_df, x='timestamp', y='normed_influence', color='layer',
                   color_discrete_sequence=px.colors.qualitative.Set2,
                   labels={'normed_influence':'Normalized Influence', 'timestamp':'Time'})
fig_area.update_layout(legend_title_text="Layer", yaxis=dict(tickformat=".0%"))
st.plotly_chart(fig_area, use_container_width=True)

# ---------- UI: signed influence timeseries ----------
st.markdown("## 2) Signed Influence (raw) — how each layer increased or decreased final scale")
fig_signed = px.line(raw_df, x='timestamp', y='signed_influence', color='layer')
fig_signed.update_layout(yaxis_title="Signed Influence (final_scale - no_layer_final)")
st.plotly_chart(fig_signed, use_container_width=True)

# ---------- UI: current breakdown ----------
st.markdown("## 3) Current Influence Breakdown (most recent timestamp)")
latest = raw_df[raw_df['timestamp'] == raw_df['timestamp'].max()]
if not latest.empty:
    latest_val = latest.set_index('layer')['signed_influence']
    # rank
    ranked = latest_val.sort_values(ascending=False)
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Current ranked influence")
        st.table(ranked.reset_index().rename(columns={'index':'layer', 0:'signed_influence'}))
    with col2:
        st.subheader("Current influence pie (abs proportions)")
        pie_df = latest_val.abs() / latest_val.abs().sum()
        fig_pie = px.pie(values=pie_df.values, names=pie_df.index, title="Abs Proportion of Influences")
        st.plotly_chart(fig_pie)
else:
    st.info("No recent timestamp available to show current breakdown.")

# ---------- UI: raw numbers & export ----------
st.markdown("## 4) Raw Influence Table & Export")
st.dataframe(influence_ts.head(200))

if st.button("Export influence CSV"):
    influence_ts.reset_index().to_csv(OUT_CSV, index=False)
    st.success(f"Saved influence CSV to {OUT_CSV}")

# ---------- UI: guidance ----------
st.markdown("## Guidance — How to read this")
st.write("""
- **Normalized stacked area** shows which layers dominate the *directional* influence across time (positive or negative is lost when normalized).
- **Signed influence** series show the raw signed contribution: positive means this layer increased final scale at that time, negative means it decreased it.
- The **leave-one-out** influence is computed by neutralizing a layer (setting its scale to 1.0) and measuring the change in the final scale proxy. A higher absolute value indicates greater causal influence on final sizing decisions.
- If a layer is consistently near zero, it is either inactive or not contributing.
- If Genome or Drift are near zero but you expect them to act, check whether your logging is capturing their dynamic values (they may be hard-coded or dead).
""")

st.markdown("## Next actions (recommended)")
st.write("""
1. If Capital dominates and you want physics to matter more, increase Consciousness priority for drift/genome or adjust learning rates for genome/adaptive layers.
2. If a layer oscillates and introduces noise, consider smoothing or increasing its update interval.
3. Use exported CSV to run further attribution (correlation with PnL, Granger tests, or causal analysis).
""")
