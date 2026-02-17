import os
import sys
import subprocess
import time

def run_pipeline():
    print("==========================================")
    print("   CHIMERA PROTOCOL: SYSTEM REPLICATION   ")
    print("==========================================")
    
    # 1. RUN ENGINE
    print("\n[1/2] Executing Chimera Engine...")
    try:
        # Run as a subprocess to ensure clean state
        subprocess.run([sys.executable, "chimera_protocol/chimera_engine.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Engine: {e}")
        return

    # 2. GENERATE REPORT
    print("\n[2/2] Generating Analytics Report...")
    try:
        subprocess.run([sys.executable, "chimera_protocol/tools/generate_report.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating Report: {e}")
        return

    print("\n------------------------------------------")
    print("   CYCLE COMPLETE. CHECK ROWSER.          ")
    print("------------------------------------------")

if __name__ == "__main__":
    # Ensure we are in the root directory
    if not os.path.exists("chimera_protocol"):
        print("Error: Run this script from the root 'hedge_fund_v3' directory.")
    else:
        run_pipeline()
