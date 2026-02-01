# Local Imports (Self-Contained)
try:
    from strategies.hfd_kinetic.backtesting.glass_engine import GlassEngine, _numba_rolling_fdi, _numba_rolling_hill
    from strategies.hfd_kinetic.backtesting.viz_engine import VizEngine
except ImportError:
    # Fallback if running from within the folder
    from backtesting.glass_engine import GlassEngine, _numba_rolling_fdi, _numba_rolling_hill
    from backtesting.viz_engine import VizEngine

# Windows Console Encoding Fix
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ==========================================
# 1. THE GOLDEN CONFIGURATION
# ==========================================
PARAMS = {
    "fdi_window": 30,       # Fast reaction to roughness
    "alpha_window": 100,    # Medium-term tail estimation
    "tail_cut": 0.05,       # Top 5% extreme returns
    "alpha_entry": 1.6,     # Enter if tails are fat (Levy Flight)
    "fdi_entry": 1.5,       # Enter if trend is structured
    "fdi_exit": 1.6         # Exit if market becomes random noise
}

# ==========================================
# 2. NUMBA KERNELS (Optimization Re-Use)
# ==========================================
@jit(nopython=True, nogil=True, cache=True)
def _numba_rolling_fdi(price_array, window):
    n = len(price_array)
    output = np.full(n, np.nan, dtype=np.float64)
    log_2 = np.log(2.0)
    log_2_n_minus_1 = np.log(2.0 * (window - 1.0))
    dt_sq = (1.0 / (window - 1.0)) ** 2
    
    for i in range(window, n):
        chunk = price_array[i-window:i]
        max_p = np.max(chunk)
        min_p = np.min(chunk)
        range_p = max_p - min_p
        
        if range_p <= 1e-9: 
            output[i] = 1.0; continue
            
        length = 0.0
        prev_norm = (chunk[0] - min_p) / range_p
        for j in range(1, window):
            curr_norm = (chunk[j] - min_p) / range_p
            # Fix scalar exponentiation typing
            dp = curr_norm - prev_norm
            length += np.sqrt(dt_sq + dp**2)
            prev_norm = curr_norm
            
        if length > 0: output[i] = 1.0 + (np.log(length) + log_2) / log_2_n_minus_1
        else: output[i] = 1.5
    return output

@jit(nopython=True, nogil=True, cache=True)
def _numba_rolling_hill(returns_array, window, tail_cut):
    n = len(returns_array)
    output = np.full(n, np.nan, dtype=np.float64)
    min_k = 2
    
    for i in range(window, n):
        chunk = np.abs(returns_array[i-window:i])
        chunk = chunk[chunk > 1e-9]
        if len(chunk) < 10: continue
            
        sorted_chunk = np.sort(chunk)[::-1]
        k = max(int(len(chunk) * tail_cut), min_k)
        
        tail_data = sorted_chunk[:k]
        x_min = tail_data[-1]
        if x_min <= 1e-9: continue
            
        sum_log = 0.0
        for x in tail_data: sum_log += np.log(x / x_min)
        
        output[i] = min(1.0 / (sum_log / k), 3.0) if sum_log > 0 else 2.0
    return output

# ==========================================
# 3. PRODUCTION PIPELINE
# ==========================================
def run_production_strategy(parquet_path):
    print(f"[ARCHITECT] Loading Data for Golden Config Validation...")
    
    if not os.path.exists(parquet_path):
        print(f"[ERROR] Parquet file not found at {parquet_path}")
        return

    # A. LOAD & PROCESS (Polars)
    # Ensure 'price' is aliased to 'close' if needed
    df = pl.read_parquet(parquet_path).sort("time").with_columns([
        pl.col("price").alias("close")
    ])
    
    # B. COMPUTE PHYSICS (Numba on 1m Aggregation)
    # We aggregate FIRST to match the optimization logic precisely
    print("[ARCHITECT] Aggregating to 1-Minute Bars...")
    df_1m = df.with_columns([
        pl.col("time").cast(pl.Datetime("ms")).alias("datetime")
    ]).group_by_dynamic("datetime", every="1m").agg([
        pl.col("close").last()
    ]).sort("datetime")
    
    close = df_1m["close"].to_numpy()
    
    # Robust Returns Calculation (1D Safe)
    # prepend=close[0] ensures length match
    returns = np.concatenate((np.array([0.0]), np.diff(close))) / (close + 1e-9)
    # Make sure we didn't shift by 1 inadvertently compared to optimization?
    # Optimization used: np.concatenate((np.array([0.0]), np.diff(close_1d)))
    # This matches.
    
    print(f"[ARCHITECT] Computing Physics Indicators (Window={PARAMS['fdi_window']})...")
    fdi = _numba_rolling_fdi(close, PARAMS["fdi_window"])
    alpha = _numba_rolling_hill(returns, PARAMS["alpha_window"], PARAMS["tail_cut"])
    
    # C. GENERATE SIGNALS
    entries = (alpha < PARAMS["alpha_entry"]) & (fdi < PARAMS["fdi_entry"])
    exits = fdi > PARAMS["fdi_exit"]
    
    # D. STRESS TEST (Monte Carlo Slippage)
    # We simulate 100 different realities where execution slippage varies randomly
    # Model: Uniform distribution between 0 bps and 5 bps per trade
    print("[ARCHITECT] Running Monte Carlo Execution Stress Test (100 Runs)...")
    
    n_sims = 100
    # Randomized slippage for each run
    monte_carlo_slippage = np.random.uniform(0.0000, 0.0005, n_sims).reshape(1, -1)
    
    # Manually broadcast signals to (Time, n_sims)
    # entries is 1D boolean array. We tile it to 100 columns.
    entries_2d = np.tile(entries.reshape(-1, 1), (1, n_sims))
    exits_2d = np.tile(exits.reshape(-1, 1), (1, n_sims))
    
    # Broadcast slippage and close to full (Time, n_sims) matrix to avoid VBT guessing
    slippage_2d = np.tile(monte_carlo_slippage, (len(close), 1))
    close_2d = np.tile(close.reshape(-1, 1), (1, n_sims))
    
    # Base Portfolio
    pf = vbt.Portfolio.from_signals(
        close=close_2d,
        entries=entries_2d, 
        exits=exits_2d,     
        fees=0.0004,
        slippage=slippage_2d,
        freq='1min',
        init_cash=100000
    )
    
    # Save Output
    print("[ARCHITECT] Generating Visualization...")
    # Plot all 100 paths (Equity Curves)
    # Uses .vbt.plot() on the DataFrame of values
    fig = pf.value().vbt.plot(title="Monte Carlo Stress Test: 100 Execution Realities")
    output_html = "monte_carlo_sim.html"
    fig.write_html(output_html)
    print(f"[ARCHITECT] Chart saved to '{output_html}'")
    
    # Report Aggregated Stats
    returns = pf.total_return() * 100
    sharpes = pf.sharpe_ratio()
    
    print("\n" + "="*50)
    print("ARCHITECT: MONTE CARLO RESULTS (100 RUNS)")
    print("="*50)
    print(f"Mean Return:       {returns.mean():.2f}%")
    print(f"Min Return:        {returns.min():.2f}% (Worst Case)")
    print(f"Max Return:        {returns.max():.2f}% (Best Case)")
    print(f"Mean Sharpe:       {sharpes.mean():.2f}")
    print(f"Prob. Profit:      {(returns > 0).mean() * 100:.1f}%")
    print("\n" + "="*50)
    print("ARCHITECT: INVESTMENT MEMORANDUM (AGGREGATE)")
    print("Strategy:  KINETIC REGIME (GOLD)")
    print("="*50)
    # Use Mean for the final summary
    print(f"Avg Total Return:  {pf.total_return().mean()*100:.2f}%")
    print(f"Avg Sharpe Ratio:  {pf.sharpe_ratio().mean():.2f}")
    print(f"Avg Max Drawdown:  {pf.max_drawdown().mean()*100:.2f}%")
    print(f"Avg Win Rate:      {pf.stats()['Win Rate [%]'].mean():.2f}%")
    print("-" * 50)
    
    # Sensitivity Check
    if pf.sharpe_ratio().mean() > 2.0:
        print("[VERDICT] SYSTEM READY FOR DEPLOYMENT.")
        print("Recommendation: Allocate 10% Risk Capital.")
    else:
        print("[VERDICT] SYSTEM UNSTABLE. DO NOT DEPLOY.")
        
    return pf

# ==========================================
# EXECUTE
# ==========================================
if __name__ == "__main__":
    parquet_file = "data/raw_crypto/ETHUSDT-trades-2023-01.parquet"
    run_production_strategy(parquet_file)
