# ==========================================================
# 🔥 CHIMERA REALITY HEATMAP
# ==========================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

TRADE_FILE = "data/chimera_blackbox_final.csv"

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

df = pd.read_csv(TRADE_FILE)

print("Loaded trades:", len(df))
print("Columns:", df.columns.tolist())

# ----------------------------------------------------------
# DETECT PNL COLUMN
# ----------------------------------------------------------

PNL_COL = None

for c in df.columns:
    if "net_pnl" in c.lower():
        PNL_COL = c

if PNL_COL is None:
    raise ValueError("No net_pnl column detected.")

df["pnl_used"] = df[PNL_COL]

# ----------------------------------------------------------
# DETECT GOVERNANCE SCALES AUTOMATICALLY
# ----------------------------------------------------------

layer_cols = [
    c for c in df.columns
    if any(k in c.lower() for k in
        ["genome","family","adaptive","meta","capital","drift","diagnostic","final"])
]

# keep numeric only
layer_cols = [
    c for c in layer_cols
    if np.issubdtype(df[c].dtype, np.number)
]

print("Detected layer columns:", layer_cols)

if len(layer_cols)==0:
    print("WARNING: No governance layers found in CSV columns. Heatmap might be empty.")
    # Allow proceeding with just PnL correlation if no layers found (just to prevent crash)
else:
    # ----------------------------------------------------------
    # BUILD REALITY MATRIX
    # ----------------------------------------------------------

    matrix = {}

    for col in layer_cols:

        corr = df[col].corr(df["pnl_used"])

        stability = df[col].std()

        matrix[col] = {
            "corr_with_pnl": corr,
            "volatility": stability
        }

    heat_df = pd.DataFrame(matrix).T

    print("\nReality Matrix:")
    print(heat_df)

    # ----------------------------------------------------------
    # 🔥 HEATMAP VISUALIZATION
    # ----------------------------------------------------------

    plt.figure(figsize=(10,6))

    sns.heatmap(
        heat_df,
        annot=True,
        cmap="coolwarm",
        center=0
    )

    plt.title("Chimera Reality Heatmap — Layer Influence vs PnL")
    plt.savefig("chimera_live/reality_heatmap.png")
    print("Reality Heatmap complete. Plot saved to chimera_live/reality_heatmap.png")
