import pandas as pd
import numpy as np
from pathlib import Path
import datetime

# Mimic the logic from reality_control_flow.py
LOG_DIR = Path("logs")
STABILITY_FILE = LOG_DIR / "stability_log.csv"
OUT_CSV = LOG_DIR / "reality_control_flow_output.csv"

def load_stability_csv():
    if not STABILITY_FILE.exists():
        return None
    try:
        df = pd.read_csv(STABILITY_FILE)
        return df
    except Exception:
        return None

def ensure_columns(df, expected_cols):
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for e in expected_cols:
        if e in cols:
            mapping[e] = cols[e]
        else:
            candidates = [c for c in df.columns if e in c.lower()]
            mapping[e] = candidates[0] if candidates else None
    return mapping

def compute_final_scale_from_row(row):
    if 'final_scale' in row and not pd.isna(row['final_scale']):
        return float(row['final_scale'])
    parts = []
    for k in ['genome_scale','drift_scale','capital_scale','adaptive_scale','meta_scale']:
        if k in row and not pd.isna(row[k]):
            try:
                parts.append(float(row[k]))
            except:
                pass
    if parts:
        return float(np.prod(parts) ** (1.0/len(parts)))
    return 1.0

def neutralize_scale(row, key):
    r = row.copy()
    r[key] = 1.0
    return r

def compute_influence_timeseries(df, layer_keys):
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
            influences[k] = float(base_final - neutral_final)
        row_out = {'index': idx, 'timestamp': r.get('timestamp', r.get('time', r.get('date', None))), 'base_final': base_final}
        row_out.update(influences)
        out_rows.append(row_out)
    return pd.DataFrame(out_rows)

if __name__ == "__main__":
    print("Loading data...")
    df = load_stability_csv()
    if df is None:
        print("Failed to load data")
        exit(1)
        
    print(f"Loaded {len(df)} rows")
    
    # normalize column names
    df.columns = [c.lower() for c in df.columns]

    expected = ['genome_scale','drift_scale','capital_scale','adaptive_scale','meta_scale','final_scale']
    mapping = ensure_columns(df, expected)

    renames = {}
    for k, v in mapping.items():
        if v and v != k:
            renames[v] = k
    df = df.rename(columns=renames)

    for k in expected:
        if k not in df.columns:
            df[k] = np.nan

    layer_keys = [k for k in expected if k.endswith('_scale') and k != 'final_scale']
    print(f"Computing influence for layers: {layer_keys}")
    
    influence_ts = compute_influence_timeseries(df, layer_keys)
    
    influence_ts.to_csv(OUT_CSV, index=False)
    print(f"Saved influence CSV to {OUT_CSV}")
    
    # Print summary stats for report
    print("\n--- INFLUENCE SUMMARY (AVG ABS VALUE) ---")
    summary = influence_ts[layer_keys].abs().mean().sort_values(ascending=False)
    print(summary)
