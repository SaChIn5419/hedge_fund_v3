# ==========================================================
# CHIMERA REALITY TEST SUITE
# ==========================================================

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

TRADE_FILE = "data/chimera_blackbox_final.csv"
FRICTION_FILE = "data/chimera_friction_analysis.csv"
GENOME_FILE = "data/genome.json"

INITIAL_CAPITAL = 1_000_000


# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

trades = pd.read_csv(TRADE_FILE)

print("Loaded trades:", len(trades))
print("Columns:", trades.columns.tolist())

# Detect pnl column
PNL_COL = None
for c in trades.columns:
    if "net_pnl" in c.lower():
        PNL_COL = c

if PNL_COL is None:
    raise ValueError("No net_pnl column found.")

trades["pnl_used"] = trades[PNL_COL]

# ----------------------------------------------------------
# BUILD BASELINE EQUITY CURVE
# ----------------------------------------------------------

def build_equity(df, pnl_col):

    daily = df.groupby("date")[pnl_col].sum()

    returns = daily / INITIAL_CAPITAL

    equity = (1 + returns).cumprod()

    return equity


baseline_eq = build_equity(trades, "pnl_used")

# ----------------------------------------------------------
# 1 GOVERNANCE OFF TEST
# ----------------------------------------------------------

def governance_off(df):

    df = df.copy()

    # Detect scale columns automatically
    scale_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in
            ["meta","capital","adaptive","final","diagnostic"])
    ]

    scale_cols = [c for c in scale_cols if np.issubdtype(df[c].dtype, np.number)]

    if len(scale_cols)==0:
        print("[WARN] No governance scale columns found.")
        return None

    df["composite_scale"] = df[scale_cols].prod(axis=1)

    df["gross_estimate"] = df["pnl_used"] / df["composite_scale"].replace({0:np.nan})

    return build_equity(df, "gross_estimate")


gov_off_eq = governance_off(trades)


# ----------------------------------------------------------
# 2 GENOME FREEZE TEST
# ----------------------------------------------------------

def genome_freeze(df):

    if "family_scale" not in df.columns:
        print("[WARN] No family_scale column -> genome freeze skipped.")
        return None

    df = df.copy()

    df["frozen_pnl"] = df["pnl_used"] / df["family_scale"].replace({0:np.nan})

    return build_equity(df, "frozen_pnl")


genome_eq = genome_freeze(trades)


# ----------------------------------------------------------
# 3 FRICTION TEST
# ----------------------------------------------------------

def friction_test(df):

    try:
        fr = pd.read_csv(FRICTION_FILE)
    except:
        print("[WARN] No friction file.")
        return None

    # Need a join key - timestamp or trade_id
    join_key = None

    for k in ["trade_id","timestamp","date"]:
        if k in df.columns and k in fr.columns:
            join_key = k

    if join_key is None:
        print("[WARN] Cannot join friction file automatically.")
        return None

    merged = df.merge(fr, on=join_key)

    # Detect slippage
    slip_col = None
    for c in merged.columns:
        if "slipp" in c.lower():
            slip_col = c

    if slip_col is None:
        print("[WARN] No slippage column detected.")
        return None

    merged["pnl_after_friction"] = merged["pnl_used"] - merged[slip_col]

    return build_equity(merged, "pnl_after_friction")


friction_eq = friction_test(trades)

# ----------------------------------------------------------
# PLOT COMPARISON
# ----------------------------------------------------------

plt.figure(figsize=(12,6))

baseline_eq.plot(label="Baseline")

if gov_off_eq is not None:
    gov_off_eq.plot(label="Governance OFF")

if genome_eq is not None:
    genome_eq.plot(label="Genome Freeze")

if friction_eq is not None:
    friction_eq.plot(label="After Friction")

plt.legend()
plt.title("Chimera Reality Comparison")
plt.savefig("chimera_live/reality_test_result.png")
print("Reality tests complete. Plot saved to chimera_live/reality_test_result.png")
