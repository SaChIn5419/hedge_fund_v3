import numpy as np
import pandas as pd
import scipy.stats as stats
import sys

# Windows Console Encoding Fix
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

class VizEngine:
    def __init__(self, simulation_results, time_index):
        """
        :param simulation_results: Dict from GlassEngine {'equity_curve': np.array, 'trades': pd.DataFrame}
        :param time_index: The DatetimeIndex corresponding to the equity curve (from the input dataframe)
        """
        self.equity = pd.Series(simulation_results['equity_curve'], index=time_index)
        self.trades = simulation_results['trades']
        self.equity.name = "Equity"
        
        # Calculate Returns (1-Minute or Tick resolution based on input)
        self.returns = self.equity.pct_change().fillna(0)
        
    def _calculate_drawdown(self):
        """Calculates Drawdown Series and Max Drawdown"""
        rolling_max = self.equity.cummax()
        drawdown = (self.equity - rolling_max) / rolling_max
        return drawdown

    def _annualization_factor(self):
        """Determines scaling factor based on data frequency"""
        # Infer frequency from time index
        if len(self.equity) < 2: return 252 # Default
        
        seconds = (self.equity.index[-1] - self.equity.index[0]).total_seconds()
        total_bars = len(self.equity)
        
        if seconds == 0: return 252

        # Bars per year approx
        seconds_per_year = 365.25 * 24 * 3600
        bars_per_year = total_bars / (seconds / seconds_per_year)
        return bars_per_year

    def report(self):
        """Generates the Institutional Tear Sheet"""
        
        # 1. RISK METRICS (The Defense)
        dd_series = self._calculate_drawdown()
        max_dd = dd_series.min()
        
        ann_factor = self._annualization_factor()
        
        # Resample to Daily for robust Sharpe calculation (Standard Practice)
        # Using 1-minute Sharpe often inflates values to 15+ due to low vol in quiet minutes
        # We handle case where re-sampling might result in empty data (short simulation)
        try:
            daily_equity = self.equity.resample('1D').last().dropna()
            if len(daily_equity) < 2:
                # Fallback for short simulations: Use raw returns normalized
                daily_ret = self.equity.pct_change().fillna(0)
                # Rescaling factor adjustment? 
                # Ideally we want Annualized Vol.
                # If 1 min bars: Ann Vol = Std * Sqrt(525600)
                ann_vol = self.returns.std() * np.sqrt(ann_factor)
                ann_ret = self.returns.mean() * ann_factor
            else:
                daily_ret = daily_equity.pct_change().fillna(0)
                ann_vol = daily_ret.std() * np.sqrt(365)
                ann_ret = daily_ret.mean() * 365
        except Exception:
             # Fallback
             daily_ret = self.returns
             ann_vol = daily_ret.std() * np.sqrt(ann_factor)
             ann_ret = daily_ret.mean() * ann_factor

        risk_free = 0.04 # 4% Risk Free Rate assumption
        
        sharpe = (ann_ret - risk_free) / ann_vol if ann_vol != 0 else 0
        
        # Sortino (Downside Volatility only)
        neg_rets = daily_ret[daily_ret < 0]
        sortino = (ann_ret - risk_free) / (neg_rets.std() * np.sqrt(365)) if len(neg_rets) > 0 else 0
        
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        # 2. TRADE METRICS (The Offense)
        if len(self.trades) > 0:
            win_rate = len(self.trades[self.trades['pnl_pct'] > 0]) / len(self.trades)
            gross_profit = self.trades[self.trades['pnl_pct'] > 0]['pnl_pct'].sum()
            gross_loss = abs(self.trades[self.trades['pnl_pct'] <= 0]['pnl_pct'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
            
            avg_win = self.trades[self.trades['pnl_pct'] > 0]['pnl_pct'].mean()
            avg_loss = self.trades[self.trades['pnl_pct'] <= 0]['pnl_pct'].mean()
            payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0
            
            # System Quality Number (SQN) - Van Tharp's Metric
            # SQN = sqrt(N) * Average Trade / Std Dev of Trades
            sqn = (np.sqrt(len(self.trades)) * self.trades['pnl_pct'].mean()) / self.trades['pnl_pct'].std()
            
            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            
            # Consecutive Losses
            # Convert wins/losses to +1/-1 series
            wins = (self.trades['pnl_pct'] > 0).astype(int)
            # Group consecutive values
            # Logic: Identify runs of 0 (losses)
            # Invert: Loss=1, Win=0
            is_loss = (self.trades['pnl_pct'] <= 0)
            # Groupby consecutive True
            cons_losses = is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumsum()
            # Only keep counts where is_loss is True
            max_consecutive_losses = cons_losses[is_loss].max()
            if pd.isna(max_consecutive_losses): max_consecutive_losses = 0

        else:
            win_rate, profit_factor, sqn, max_consecutive_losses, expectancy, payoff_ratio = 0, 0, 0, 0, 0, 0

        # 3. PRINT REPORT
        print("\n" + "="*50)
        print("VIZ ENGINE: INSTITUTIONAL TEAR SHEET")
        print("="*50)
        print(f"Time Range:       {self.equity.index[0]} -> {self.equity.index[-1]}")
        print(f"Data Frequency:   {ann_factor:.0f} bars/year")
        print("-" * 50)
        print(">>> PERFORMANCE")
        print(f"Total Return:     {((self.equity.iloc[-1]/self.equity.iloc[0])-1)*100:.2f}%")
        print(f"CAGR (Ann Ret):   {ann_ret*100:.2f}%")
        print(f"Volatility (Ann): {ann_vol*100:.2f}%")
        print("-" * 50)
        print(">>> RISK ADJUSTED")
        print(f"Sharpe Ratio:     {sharpe:.2f} (Benchmark: >1.0)")
        print(f"Sortino Ratio:    {sortino:.2f} (Benchmark: >1.5)")
        print(f"Calmar Ratio:     {calmar:.2f} (Benchmark: >2.0)")
        print(f"Max Drawdown:     {max_dd*100:.2f}%")
        print("-" * 50)
        print(">>> TRADE MECHANICS")
        print(f"Trade Count:      {len(self.trades)}")
        print(f"Win Rate:         {win_rate*100:.2f}%")
        print(f"Profit Factor:    {profit_factor:.2f}")
        print(f"Payoff Ratio:     {payoff_ratio:.2f} (Avg Win / Avg Loss)")
        print(f"Expectancy:       {expectancy*100:.2f}% per trade")
        print(f"SQN Score:        {sqn:.2f}")
        print(f"Max Cons. Loss:   {max_consecutive_losses}")
        print("="*50)
        
        return dd_series
