import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio

class ArchitectVolumeProfile:
    def __init__(self, ticker, lookback_days=60, bins=50):
        self.ticker = ticker
        self.lookback = lookback_days
        self.bins = bins
        
    def fetch_data(self):
        # Fetch slightly more data to ensure coverage
        print(f"AMT: Fetching {self.ticker}...")
        df = yf.download(self.ticker, period="6mo", progress=False)
        
        # Flatten Multi-Index if necessary (The YFinance Fix)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Standardize Columns
        if 'Close' not in df.columns and len(df.columns) == 5:
             # Try assuming standard OHLCV
             df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Slice the last N days for the Profile
        self.df = df.iloc[-self.lookback:].copy()
        return self.df

    def calculate_profile(self):
        """
        Calculates the Price-by-Volume Histogram.
        """
        # Ensure we have data
        if not hasattr(self, 'df') or self.df.empty:
            self.fetch_data()
            
        # 1. Define Price Bins (from Low to High)
        min_price = self.df['Low'].min()
        max_price = self.df['High'].max()
        price_bins = np.linspace(min_price, max_price, self.bins)
        
        # 2. Bucket Volume into Bins
        # We assume volume is distributed evenly across the candle's range (Approximation)
        # A more precise way for Daily data is to attribute volume to the Typical Price.
        self.df['Typical_Price'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        
        # Digitize: Find which bin each day's Typical Price belongs to
        bin_indices = np.digitize(self.df['Typical_Price'], price_bins)
        
        # Sum volume per bin
        profile_volume = np.zeros(len(price_bins))
        for i, vol in enumerate(self.df['Volume']):
            bin_idx = bin_indices[i] - 1 # Adjust 0-based
            if 0 <= bin_idx < len(profile_volume):
                profile_volume[bin_idx] += vol
                
        # 3. Calculate Value Area (70% of Volume)
        total_volume = np.sum(profile_volume)
        target_volume = 0.70 * total_volume
        
        # Find Point of Control (POC) - The bin with Max Volume
        poc_idx = np.argmax(profile_volume)
        poc_price = price_bins[poc_idx]
        
        # Expand out from POC to find VAH and VAL
        current_volume = profile_volume[poc_idx]
        upper_idx = poc_idx
        lower_idx = poc_idx
        
        while current_volume < target_volume:
            # Try adding above
            vol_up = 0
            if upper_idx < len(profile_volume) - 1:
                vol_up = profile_volume[upper_idx + 1]
                
            # Try adding below
            vol_down = 0
            if lower_idx > 0:
                vol_down = profile_volume[lower_idx - 1]
            
            # Add the larger neighbor
            if vol_up > vol_down:
                upper_idx += 1
                current_volume += vol_up
            else:
                lower_idx -= 1
                current_volume += vol_down
                
            # Break if we hit boundaries
            if lower_idx == 0 and upper_idx == len(profile_volume) - 1:
                break
                
        vah = price_bins[upper_idx] # Value Area High
        val = price_bins[lower_idx] # Value Area Low
        
        return {
            'bins': price_bins,
            'volume': profile_volume,
            'POC': poc_price,
            'VAH': vah,
            'VAL': val,
            'Current_Price': self.df['Close'].iloc[-1]
        }

    def visualize(self):
        metrics = self.calculate_profile()
        
        fig = go.Figure()
        
        # 1. Price Candle Chart
        fig.add_trace(go.Candlestick(
            x=self.df.index,
            open=self.df['Open'], high=self.df['High'],
            low=self.df['Low'], close=self.df['Close'],
            name='Price'
        ))
        
        # 2. Volume Profile (Horizontal Bars)
        # We scale volume to fit on the X-axis time scale (Hack for visualization)
        max_vol = np.max(metrics['volume'])
        time_range = (self.df.index.max() - self.df.index.min()).days
        # Scaling factor to make bars visible but not overwhelming
        scale = (time_range * 0.3) / max_vol 
        
        # Create horizontal bars using Shapes
        # We start drawing from the right-most timestamp backwards
        start_date = self.df.index.max()
        
        shapes = []
        for i, vol in enumerate(metrics['volume']):
            bar_len_days = vol * scale
            # We assume day delta approx
            if bar_len_days > 0:
                 end_date_time = start_date - pd.Timedelta(days=int(bar_len_days))
                 
                 shapes.append(dict(
                    type="line",
                    x0=start_date, y0=metrics['bins'][i],
                    x1=end_date_time, y1=metrics['bins'][i],
                    line=dict(color="rgba(0, 0, 255, 0.3)", width=2),
                 ))

        fig.update_layout(shapes=shapes)

        # 3. Key Levels (POC, VAH, VAL)
        fig.add_hline(y=metrics['POC'], line_color="red", line_width=2, annotation_text="POC (Equilibrium)")
        fig.add_hline(y=metrics['VAH'], line_color="green", line_width=1, annotation_text="VAH (Breakout Line)")
        fig.add_hline(y=metrics['VAL'], line_color="green", line_width=1, annotation_text="VAL (Support)")
        
        # 4. Assessment
        cp = metrics['Current_Price']
        if cp > metrics['VAH']:
            status = "BULLISH BREAKOUT (Vacuum)"
            color = "green"
        elif cp < metrics['VAL']:
            status = "BEARISH BREAKDOWN"
            color = "red"
        else:
            status = "TRAPPED IN VALUE (Chop)"
            color = "orange"
            
        fig.update_layout(
            title=f"ARCHITECT VOLUME PROFILE: {self.ticker} | Status: {status}",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            font=dict(color=color)
        )
        
        filename = f"vp_{self.ticker.replace('^', '')}.html"
        fig.write_html(filename)
        print(f"Generated: {filename} | State: {status}")
        print(f"Levels -> POC: {metrics['POC']:.2f} | VAH: {metrics['VAH']:.2f} | VAL: {metrics['VAL']:.2f}")

# --- EXECUTION ---
if __name__ == "__main__":
    # Test on Nifty
    vp = ArchitectVolumeProfile('^NSEI', lookback_days=90)
    vp.visualize()
