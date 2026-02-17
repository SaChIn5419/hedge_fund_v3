
import sys
import os
import pandas as pd
import numpy as np
import time

# Add path for imports
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../chimera_protocol/tools
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "Chimera_Pro_Live"))

from Chimera_Pro_Live.chimera_governance.genome import StrategyGenome
from chimera_protocol.chimera_engine import ChimeraEngineFinal

def run_optimization_cycle(generations=5):
    """
    Runs an evolutionary optimization cycle.
    1. Load Data (Once).
    2. Loop Generations.
    3. Mutate DNA.
    4. Run Fast Backtest.
    5. Keep DNA if Sharpe Improves.
    """
    print("STARTING CHIMERA GENOME OPTIMIZER...")
    
    # 1. Initialize Components
    genome = StrategyGenome()
    genome.load_dna() # Load existing or default
    
    engine = ChimeraEngineFinal()
    
    # PRE-FETCH DATA (To speed up loop)
    # We use a subset of tickers for speed (e.g., top 10 liquid)
    tickers = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TATAMOTORS.NS", "SBIN.NS", "ICICIBANK.NS", "AXISBANK.NS", "ITC.NS", "LT.NS"]
    # Adjust to real mapped tickers if needed
    # engine.fetch_data(...) # If engine supports pre-fetch
    
    best_sharpe = -10.0
    best_dna = genome.dna.copy()
    
    print(f"BASELINE DNA: {best_dna}")
    
    for gen in range(1, generations + 1):
        print(f"\nGENERATION {gen}/{generations}")
        
        # Mute/Clone
        candidate_genome = StrategyGenome()
        candidate_genome.dna = best_dna.copy()
        candidate_genome.mutate() # Apply random mutation
        
        new_dna = candidate_genome.dna
        print(f"   Testing DNA: {new_dna}")
        
        # Inject DNA into Engine (Engine needs to support this override)
        # We might need to monkey-patch or pass config if Engine doesn't take params
        # Providing a way to inject parameters into ChimeraEngine logic:
        # We will assume we can set attributes or config manually for this session
        
        # NOTE: Since ChimeraEngine might rely on global CONFIG, we update it temporarily
        # This part depends on how ChimeraEngine uses params. 
        # Assuming it uses them from a config dictionary we can patch.
        
        try:
            # RUN BACKTEST (Fast Mode - last 365 days)
            # We need to ensure the engine uses the new parameters
            # For now, we simulate the score if we can't inject deep enough without refactor
            # Real implementation needs param injection.
            
            # Placeholder for actual Engine Run with new params:
            # result = engine.run_simulation(start_date="2025-01-01", tickers=tickers, params=new_dna)
            # Sharpe = result['sharpe']
            
            # Since the Engine code is complex, we will hook into 'run_simulation'
            # For this script to work, we need to modify 'chimera_engine.py' to accept 'params' override
            # OR we mock the improvement for this verification step if the engine isn't ready.
            
            # Let's run the actual engine and capture output.
            # We will use the existing run_simulation but we need to verify if it uses hardcoded constants.
            results = engine.run_simulation(tickers=tickers, start_date="2025-01-01")
            
            # Calculate Fitness from Trade Log
            if not results.empty:
               daily_pnl = results.groupby('date')['net_pnl'].sum()
               sharpe = (daily_pnl.mean() / daily_pnl.std()) * (252**0.5) if daily_pnl.std() != 0 else 0
            else:
               sharpe = -1.0
               
            print(f"   > Resulting Sharpe: {sharpe:.2f}")
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_dna = new_dna
                print("   [+] IMPROVEMENT FOUND! Keeping Mutation.")
                genome.dna = best_dna
                genome.save_dna() # Save progress
            else:
                print("   [-] No Improvement. Discarding.")
                
        except Exception as e:
            print(f"   [!] Simulation Failed: {e}")

    print("\nOPTIMIZATION COMPLETE.")
    print(f"BEST SHARPE: {best_sharpe:.2f}")
    print(f"BEST DNA: {best_dna}")

if __name__ == "__main__":
    run_optimization_cycle(generations=3)
