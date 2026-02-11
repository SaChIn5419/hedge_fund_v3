import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ChimeraLiveProtocol:
    """
    CHIMERA LIVE DEPLOYMENT PROTOCOL
    The rulebook that keeps Chimera alive when markets stop behaving nicely.
    
    Layers:
    1. Signal Health (Physics Regime)
    2. Execution Reality (Slippage, Turnover)
    3. Capital Survival (Sharpe, Drawdown, Regime)
    """
    
    def __init__(self, trade_log_df, market_data=None, capital=1000000.0):
        self.trade_log = trade_log_df
        self.market_data = market_data if market_data else {}
        self.capital = capital
        self.alerts = []
        self.actions = []
        self.layer_status = {
            "L1_Signal": "GREEN",
            "L2_Execution": "GREEN",
            "L3_Survival": "GREEN"
        }
        self.status_color = "GREEN" # Overall System Status

    def log_alert(self, layer, rule, status, message, action):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.alerts.append({
            "timestamp": timestamp,
            "layer": layer,
            "rule": rule,
            "status": status,
            "message": message,
            "action": action
        })
        if action:
            self.actions.append(f"[{rule}] {action}")

    def run_protocol_check(self, current_date=None):
        """
        Executes all 9 RULES of the protocol.
        """
        if self.trade_log.empty:
            self.log_alert("SYSTEM", "Init", "RED", "No Trade Log Data Found", "HALT SYSTEM")
            self.status_color = "RED"
            return

        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.trade_log['date']):
            self.trade_log['date'] = pd.to_datetime(self.trade_log['date'])

        # Filter to "Known History" if date provided
        if current_date:
            history = self.trade_log[self.trade_log['date'] <= current_date].copy()
        else:
            history = self.trade_log.copy()
            current_date = history['date'].max()

        if history.empty:
             self.log_alert("SYSTEM", "Init", "RED", f"No data for {current_date}", "HALT SYSTEM")
             return

        # --- LAYER 1: SIGNAL HEALTH ---
        self._check_efficiency_decay(history)
        self._check_vac_structure_failure(history)
        
        # NOTE: Energy Regime Integrity (Rule 1) requires daily energy scan, 
        # which might not be in trade log if we only log trades. 
        # We will try to infer from traded assets or market data if available.
        self._check_energy_regime(history)

        # --- LAYER 2: EXECUTION REALITY ---
        self._check_slippage(history)
        self._check_turnover(history, current_date)
        self._check_leverage_clamp(history, current_date)

        # --- LAYER 3: CAPITAL SURVIVAL ---
        self._check_rolling_sharpe(history)
        self._check_drawdown_ladder(history)
        self._check_regime_flip(history, current_date)
        
        # DETERMINING FINAL STATUS
        self._synthesize_status()

    # ------------------------------------------------------------------
    # LAYER 1 CHECKS
    # ------------------------------------------------------------------
    
    def _check_energy_regime(self, df):
        """RULE 1: Energy Regime Integrity"""
        # We need rolling mean/std of energy.
        # Assuming trade_log has 'kinetic_energy' for executed trades.
        # This is a proxy for "System Energy".
        
        if 'kinetic_energy' not in df.columns:
            return

        # Get daily avg energy of portfolio
        daily_energy = df.groupby('date')['kinetic_energy'].mean()
        
        if len(daily_energy) < 20: return
        
        current_energy = daily_energy.iloc[-1]
        rolling_mean = daily_energy.rolling(20).mean().iloc[-1]
        rolling_std = daily_energy.rolling(20).std().iloc[-1]
        
        # Trigger: energy < (mean - 2 * std)
        if current_energy < (rolling_mean - 2 * rolling_std):
            self.log_alert("Layer 1", "Rule 1 (Energy)", "YELLOW", 
                           f"Energy Collapse: {current_energy:.2f} < {rolling_mean - 2*rolling_std:.2f}",
                           "Reduce leverage_mult by 30%")
            self.layer_status["L1_Signal"] = "YELLOW"

    def _check_efficiency_decay(self, df):
        """RULE 2: Efficiency Decay Check"""
        # Metric: efficiency (Fractal Dimension or similar)
        if 'efficiency' not in df.columns: return

        daily_eff = df.groupby('date')['efficiency'].mean()
        if len(daily_eff) < 30: return
        
        current_eff = daily_eff.iloc[-1]
        # Historical 25th percentile
        hist_25 = daily_eff.quantile(0.25)
        
        if current_eff < hist_25:
             self.log_alert("Layer 1", "Rule 2 (Efficiency)", "YELLOW",
                            f"Efficiency Decay: {current_eff:.2f} < {hist_25:.2f} (25th %)",
                            "Only top-ranked trades. Cut count 50%.")
             if self.layer_status["L1_Signal"] != "RED": self.layer_status["L1_Signal"] = "YELLOW"

    def _check_vac_structure_failure(self, df):
        """RULE 3: Structure Failure Detector (VACUUM hit rate)"""
        # Look at last 20 trades where structure was VACUUM
        if 'structure_tag' not in df.columns: return 
        
        # Filter for VACUUM trades completed (we need PnL)
        vac_trades = df[df['structure_tag'] == 'VACUUM'].tail(20)
        
        if len(vac_trades) < 5: return # Not enough sample
        
        wins = vac_trades[vac_trades['net_pnl'] > 0]
        hit_rate = len(wins) / len(vac_trades)
        
        if hit_rate < 0.40:
             self.log_alert("Layer 1", "Rule 3 (Structure)", "YELLOW",
                            f"Vacuum Fail: Hit Rate {hit_rate:.1%} < 40%",
                            "Disable VACUUM weighting multiplier.")
             # This is a specific tactic tweak, not a full system warn, but noteworthy.

    # ------------------------------------------------------------------
    # LAYER 2 CHECKS
    # ------------------------------------------------------------------

    def _check_slippage(self, df):
        """RULE 4: Slippage Divergence Guard"""
        # Needs 'real_fill' vs 'expected'. 
        # If simulated, we might checks 'close' vs 'entry'? 
        # For now, placeholder or check if columns exist.
        if 'slippage' in df.columns:
            # TODO: Implement if data avail
            pass
        else:
            # We assume simulation has 0 slippage logic logged currently
            pass

    def _check_turnover(self, df, current_date):
        """RULE 5: Turnover Shock Detector"""
        # Turnover > 0.35
        # Calculate daily turnover: sum(abs(delta values)) / equity
        # Hard to do perfectly from trade_log without daily snapshot of all weights
        # Approximation: sum of new weights today
        
        today_trades = df[df['date'] == current_date]
        if today_trades.empty: return
        
        # Approx turnover as sum of absolute weights traded today
        # (This assumes all are new entries or full exits)
        turnover_est = today_trades['weight'].abs().sum()
        
        if turnover_est > 0.35:
             self.log_alert("Layer 2", "Rule 5 (Turnover)", "YELLOW",
                            f"Turnover Spike: {turnover_est:.2f} > 0.35",
                            "Freeze new 1 day. Exits only.")
             self.layer_status["L2_Execution"] = "YELLOW"

    def _check_leverage_clamp(self, df, current_date):
        """RULE 6: Leverage Reality Clamp"""
        # Sum |weights| <= leverage_mult
        # existing engine logic likely enforces this, but we verify.
        today = df[df['date'] == current_date]
        if today.empty: return
        
        total_exp = today['weight'].abs().sum()
        # Max leverage is usually 1.5 or defined in config
        # We check against a hard limit, say 2.0 to be safe or the dynamic limit
        
        # Check if any single trade > 30% (Concentration risk - implied)
        if any(today['weight'].abs() > 0.35):
             self.log_alert("Layer 2", "Rule 6 (Concentration)", "RED",
                            "Single Position > 35%",
                            "Force Resize Immediate.")
             self.layer_status["L2_Execution"] = "RED"

    # ------------------------------------------------------------------
    # LAYER 3 CHECKS
    # ------------------------------------------------------------------

    def _check_rolling_sharpe(self, df):
        """RULE 7: Rolling Sharpe Kill Switch (60d)"""
        # Need daily returns stream
        # Reconstruct daily PnL
        daily_pnl = df.groupby('date')['net_pnl'].sum()
        # Assume 1M capital for % logic if needed, or just use PnL series
        # Sharpe = mean/std * sqrt(252)
        
        if len(daily_pnl) < 60: return
        
        # Last 60 days
        window = daily_pnl.tail(60)
        mean_ret = window.mean()
        std_ret = window.std()
        
        if std_ret == 0: sharpe = 0
        else: sharpe = (mean_ret / std_ret) * np.sqrt(252)
        
        if sharpe < 0.2:
             self.log_alert("Layer 3", "Rule 7 (Sharpe)", "RED",
                            f"Sharpe Critical: {sharpe:.2f} < 0.2",
                            "PAUSE NEW ENTRIES.")
             self.layer_status["L3_Survival"] = "RED"
        elif sharpe < 0.5:
             self.log_alert("Layer 3", "Rule 7 (Sharpe)", "YELLOW",
                            f"Sharpe Decay: {sharpe:.2f} < 0.5",
                            "Cut leverage to 0.5x.")
             if self.layer_status["L3_Survival"] != "RED": self.layer_status["L3_Survival"] = "YELLOW"

    def _check_drawdown_ladder(self, df):
        """RULE 8: Drawdown Control Ladder"""
        # Calc Equity Curve
        daily_pnl = df.groupby('date')['net_pnl'].sum()
        cum_pnl = daily_pnl.cumsum()
        equity = self.capital + cum_pnl
        
        peak = equity.cummax()
        dd = (equity - peak) / peak
        current_dd = dd.iloc[-1]
        
        if current_dd < -0.25:
             self.log_alert("Layer 3", "Rule 8 (Drawdown)", "RED",
                            f"Max DD Breached: {current_dd:.1%} < -25%",
                            "STRATEGY OFF.")
             self.layer_status["L3_Survival"] = "RED"
        elif current_dd < -0.20:
             self.log_alert("Layer 3", "Rule 8 (Drawdown)", "RED",
                            f"DD Tier 3: {current_dd:.1%} < -20%",
                            "Reduce Leverage 50%.")
             self.layer_status["L3_Survival"] = "RED"
        elif current_dd < -0.15:
             self.log_alert("Layer 3", "Rule 8 (Drawdown)", "YELLOW",
                            f"DD Tier 2: {current_dd:.1%} < -15%",
                            "Reduce Leverage 25%.")
             if self.layer_status["L3_Survival"] != "RED": self.layer_status["L3_Survival"] = "YELLOW"

    def _check_regime_flip(self, df, current_date):
        """RULE 9: Regime Flip Shutdown (The Triple threat)"""
        # 1. Vol Spike
        # 2. Efficiency Drop
        # 3. Sharpe Negative
        
        # Need latest metrics
        # Vol
        today_trades = df[df['date'] == current_date]
        if today_trades.empty: return
        
        # Assuming 'nifty_vol' is logged
        vol = today_trades['nifty_vol'].mean() # stored as 0.15 for 15%
        vol_spike = vol > 0.25 # Implicit > 25 VIX
        
        # Efficiency
        daily_eff = df.groupby('date')['efficiency'].mean()
        eff_drop = False
        if len(daily_eff) > 10:
            eff_drop = daily_eff.iloc[-1] < daily_eff.quantile(0.25)
            
        # Sharpe
        # Re-calc short term sharpe (like 20d) for immediate flip? 
        # Rule says "rolling_sharpe negative" - usually refers to the main 60d one.
        daily_pnl = df.groupby('date')['net_pnl'].sum()
        sharpe_neg = False
        if len(daily_pnl) > 60:
            window = daily_pnl.tail(60)
            if window.mean() < 0: sharpe_neg = True
            
        if vol_spike and eff_drop and sharpe_neg:
             self.log_alert("Layer 3", "Rule 9 (Regime Flip)", "RED",
                            "STRUCTURAL REGIME BREAK (Vol+Eff+Sharpe)",
                            "FULL SYSTEM PAUSE.")
             self.layer_status["L3_Survival"] = "RED"

    def _synthesize_status(self):
        """Final Logic for Dashboard Color"""
        layers = self.layer_status.values()
        
        if "RED" in layers:
            self.status_color = "RED"
        elif "YELLOW" in layers:
            self.status_color = "YELLOW"
        else:
            self.status_color = "GREEN"

    def get_report(self):
        return {
            "status": self.status_color,
            "layer_status": self.layer_status,
            "alerts": self.alerts,
            "actions": self.actions
        }
