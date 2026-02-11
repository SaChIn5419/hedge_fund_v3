import os
import sys
import subprocess

def launch_monitor():
    print("==========================================")
    print("   CHIMERA LIVE CONTROL PANEL REPLICATION ")
    print("==========================================")
    
    # Path to the visualizer tool
    tool_path = os.path.join("chimera_protocol", "tools", "chimera_live_panel.py")
    
    if not os.path.exists(tool_path):
        print(f"Error: Could not find tool at {tool_path}")
        return

    print(f"\nLaunching Streamlit Dashboard: {tool_path}")
    print("Press Ctrl+C to stop.")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", tool_path], check=True)
    except KeyboardInterrupt:
        print("\nMonitor Stopped.")
    except Exception as e:
        print(f"Error launching monitor: {e}")

if __name__ == "__main__":
    if not os.path.exists("chimera_protocol"):
        print("Error: Run this script from the root 'hedge_fund_v3' directory.")
    else:
        launch_monitor()
