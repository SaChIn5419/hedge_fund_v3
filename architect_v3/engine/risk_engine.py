import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import genpareto, t
from scipy.optimize import minimize

class GarchEvtRiskEngine:
    """
    PHASE 7 RISK ENGINE: GARCH-EVT-COPULA
    Replaces static Stop Losses with Dynamic VaR Limits.
    """
    def __init__(self, confidence_level=0.99):
        self.alpha = 1 - confidence_level # 0.01 for 99% VaR
        
    def fit_garch(self, returns):
        """
        Fits GARCH(1,1) to estimate conditional volatility.
        """
        # Rescale returns for better convergence (usually x100)
        scale = 100.0
        scaled_rets = returns * scale
        
        # GARCH(1,1) with Skew Student-t distribution (captures fat tails better than Normal)
        model = arch_model(scaled_rets, vol='Garch', p=1, q=1, dist='skewt')
        res = model.fit(disp='off', show_warning=False)
        
        # Extract Conditional Volatility
        cond_vol = res.conditional_volatility / scale
        
        # Standardized Residuals
        std_resid = res.resid / res.conditional_volatility
        
        return cond_vol, std_resid, res

    def fit_evt_tails(self, std_resid, threshold_q=0.05):
        """
        Fits Generalized Pareto Distribution (GPD) to the lower tail of residuals.
        Extreme Value Theory to model Black Swan probabilities.
        """
        # Focus on left tail (losses)
        # We look at q-th quantile threshold
        threshold = np.quantile(std_resid, threshold_q)
        
        # Exceedances (Values below threshold)
        # For EVT maths, we often flip sign to handle 'excess over threshold'
        # Loss L = -R. Threshold u. Excess y = L - u.
        losses = -std_resid
        u = -threshold
        excess = losses[losses > u] - u
        
        if len(excess) < 10:
             # Fallback if not enough tail data
             return 0.2, 0.0, u # Shape, Scale, Threshold
             
        # Fit GPD
        # shape (c), loc (0), scale (s)
        # We fix loc=0 for POT method
        params = genpareto.fit(excess, floc=0)
        shape, _, scale = params
        
        return shape, scale, u

    def calculate_dynamic_var(self, returns, window=1000):
        """
        Calculates Ex-Ante Value at Risk (VaR) using GARCH-EVT.
        Optimization: Rolling Window analysis.
        """
        results = []
        
        # Need minimum history
        if len(returns) < window:
             print("Insufficient history for GARCH-EVT.")
             return None
             
        # Rolling forecast (simulated for last N days)
        # For production, we just run on the latest window to get Tomorrow's Limit
        
        # 1. Fit GARCH on window
        subset = returns.iloc[-window:]
        cond_vol, std_resid, model_res = self.fit_garch(subset)
        
        # 2. Fit EVT on Residuals
        shape, scale, u = self.fit_evt_tails(std_resid)
        
        # 3. Calculate VaR based on EVT formula
        # VaR_q = sigma_{t+1} * VaR_resid_q
        
        # Forecast next day volatility
        forecasts = model_res.forecast(horizon=1)
        next_vol_scaled = np.sqrt(forecasts.variance.values[-1, :])[0]
        next_vol = next_vol_scaled / 100.0
        
        # VaR for Standardized Residuals using GPD inverse (PPF)
        # q = alpha (e.g., 0.01)
        # N_u = Number of exceedances needed?
        # VaR_z = u + (scale/shape) * [ ((q * N) / N_u)^(-shape) - 1 ]
        # Simplified GPD VaR:
        # P(Z > z) = (1 + xi * (z-u)/sigma)^(-1/xi) * (Nu/N)
        
        # Let's use simple percentile of fitted t-dist from GARCH if EVT fails, 
        # but here we implement true EVT
        
        total_obs = len(std_resid)
        n_exceed = np.sum(-std_resid > u)
        
        if n_exceed == 0:
             # Fallback to empirical quantile
             z_var = np.quantile(std_resid, self.alpha)
        else:
             # GPD VaR Formula
             # alpha is the tail probability (0.01)
             term1 = (self.alpha * total_obs) / n_exceed
             term2 = term1 ** (-shape) - 1
             z_var_loss = u + (scale / shape) * term2
             z_var = -z_var_loss # Convert back to return sign
             
        # Dynamic VaR
        dynamic_var = next_vol * z_var
        
        return dynamic_var, next_vol

    def get_risk_report(self, df_prices):
        """
        Full Risk Scan for the Tearsheet.
        """
        returns = df_prices['Close'].pct_change().dropna()
        var_limit, vol = self.calculate_dynamic_var(returns)
        
        return {
            "GARCH_Vol_Annual": vol * np.sqrt(252),
            "VaR_99_Daily": var_limit,
            "Recc_Stop_Loss": var_limit * 1.5 # 1.5x VaR as safety
        }
