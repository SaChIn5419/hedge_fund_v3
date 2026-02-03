import polars as pl
import numpy as np
import pandas as pd

CSV_PATH = "trade_log.csv"
COLUMNS = [
    "date", "ticker", "close", "sma_20", "upper_band", "lower_band", 
    "momentum", "energy", "volatility", "rupee_volume", 
    "alpha_weight", "market_regime", "max_weight", "weight", 
    "fwd_return", "turnover", "position_value", 
    "gross_pnl", "friction_pnl", "net_pnl"
]

def run_audit():
    print("--- INSTITUTIONAL STRATEGY AUDIT (FORENSIC) ---")
    
    try:
        # Load Data with Schema Enforcement
        df = pl.read_csv(CSV_PATH, has_header=False, new_columns=COLUMNS)
        
        # Explicit Casting (Fix for String/Numeric Error)
        df = df.with_columns([
            pl.col("weight").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("fwd_return").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("gross_pnl").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("market_regime").cast(pl.Float64, strict=False).fill_null(1.0),
        ])
        
        # Filter for Active Trades
        # A "Trade" is any day where we held a position (weight > 0)
        # To count distinct "Round Trips" is harder without trade IDs, 
        # so we analyze "Daily Opportunities".
        active_days = df.filter(pl.col("weight") > 0)
        
        total_days = active_days.height
        if total_days == 0:
            print("No trades found in log.")
            return

        print(f"Total Active Trading Days (across all assets): {total_days}")
        
        # 1. FALSE POSITIVE RATE (Daily Basis)
        # False Positive = We held a position, but it lost money that day (fwd_return < 0)
        losing_days = active_days.filter(pl.col("fwd_return") < 0)
        winning_days = active_days.filter(pl.col("fwd_return") > 0)
        
        cnt_loss = losing_days.height
        cnt_win = winning_days.height
        
        win_rate = cnt_win / total_days
        false_positive_rate = cnt_loss / total_days
        
        print(f"\n[SIGNAL QUALITY]")
        print(f"Win Rate (Daily Edge):     {win_rate:.2%}")
        print(f"False Positive Rate:       {false_positive_rate:.2%} (The 'Noise' Level)")
        
        # 2. EXPECTANCY (Risk/Reward)
        avg_win = winning_days["fwd_return"].mean()
        avg_loss = losing_days["fwd_return"].mean() # Is negative
        
        profit_factor = abs(winning_days["gross_pnl"].sum() / losing_days["gross_pnl"].sum())
        
        print(f"\n[EXPECTANCY]")
        print(f"Avg Win:                   {avg_win:.2%}")
        print(f"Avg Loss:                  {avg_loss:.2%}")
        print(f"Risk/Reward Ratio:         {abs(avg_win/avg_loss):.2f}")
        print(f"Profit Factor:             {profit_factor:.2f} (Institutional Threshold > 1.5)")
        
        # 3. STREAK ANALYSIS (Ticker Level)
        # Max Consecutive Losses per Ticker
        # We need to sort by ticker and date
        # Then identify groups of consecutive negative returns
        
        print(f"\n[STREAK FORENSICS]")
        # Convert to pandas for easier shift logic on groups
        pdf = active_days.select(["ticker", "date", "fwd_return"]).to_pandas()
        pdf['is_loss'] = pdf['fwd_return'] < 0
        
        # Group by ticker, then find max run of True
        max_loss_streak = 0
        worst_ticker = ""
        
        for ticker, group in pdf.groupby('ticker'):
            # Logic: group['is_loss'] -> 0, 1, 1, 0, 1, 1, 1
            # fast groupby cumsum trick
            streaks = group['is_loss'].groupby((group['is_loss'] != group['is_loss'].shift()).cumsum()).cumsum()
            # This counts 1, 2, 3... but resets on False? No, simpler way:
            
            # Simple iteration
            current_streak = 0
            local_max = 0
            for val in group['is_loss']:
                if val:
                    current_streak += 1
                else:
                    local_max = max(local_max, current_streak)
                    current_streak = 0
            local_max = max(local_max, current_streak)
            
            if local_max > max_loss_streak:
                max_loss_streak = local_max
                worst_ticker = ticker
                
        print(f"Max Consecutive Losing Days: {max_loss_streak} ({worst_ticker})")
        
        # 4. FALSE POSITIVE DISTRIBUTION (By Regime)
        # Are we losing more in Bear Markets?
        print(f"\n[REGIME BREAKDOWN]")
        regimes = active_days.group_by("market_regime").agg([
            pl.count().alias("count"),
            pl.col("fwd_return").mean().alias("avg_ret"),
            (pl.col("fwd_return") < 0).sum().alias("losses")
        ])
        
        for row in regimes.to_dicts():
            regime_name = "BULL" if row['market_regime'] == 1.0 else "BEAR/VOL"
            fp_rate = row['losses'] / row['count']
            print(f"Regime {row['market_regime']} ({regime_name}): FP Rate = {fp_rate:.2%} | Avg Ret = {row['avg_ret']:.2%}")
            
        print("\n[VERDICT]")
        if false_positive_rate > 0.60:
            print("CRITICAL: Strategy is Noisier than Random (FP > 60%). Needs Filter Tightening.")
        elif false_positive_rate > 0.50:
             print("WARNING: Strategy is a Coin Flip (50/50). Alpha comes only from Sizing (R/R).")
        else:
             print("PASS: Signal has Predictive Edge (Win Rate > 50%).")
             
    except Exception as e:
        print(f"AUDIT FAILED: {e}")

if __name__ == "__main__":
    run_audit()
