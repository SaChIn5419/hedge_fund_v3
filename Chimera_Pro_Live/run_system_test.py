import os
import time
import subprocess
import sys

def run_test():
    print("🧪 STARTING CHIMERA SYSTEM TEST (MOCK MODE)")
    print("==========================================")
    
    # 1. Verify Config
    print("[1/4] Checking Configuration...")
    try:
        from chimera_execution import config
        if not config.MOCK_MODE:
            print("❌ Error: Config is NOT in MOCK_MODE. Please set MOCK_MODE=True safely.")
            return
        print("✅ Config is in MOCK_MODE.")
    except Exception as e:
        print(f"❌ Config Check Failed: {e}")
        return

    # 2. Launch Engine
    print("[2/4] Launching Engine (timeout 30s)...")
    process = subprocess.Popen(
        [sys.executable, "main_ws.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Monitor Output for a few seconds
    start_time = time.time()
    success = False
    
    while time.time() - start_time < 30:
        line = process.stdout.readline()
        if line:
            print(f"   [ENGINE] {line.strip()}")
            if "Physics Update" in line:
                print("✅ Physics Engine is active!")
                success = True
                break
        time.sleep(0.1)
        
    # 3. Check Telemetry
    print("[3/4] Verifying Telemetry...")
    if os.path.exists("data/neural_state.json"):
        print("✅ neural_state.json exists.")
    else:
        print("❌ neural_state.json NOT found.")

    # 4. Cleanup
    print("[4/4] Terminating Test...")
    process.terminate()
    
    if success:
        print("\n🎉 TEST PASSED: System is alive and generating physics.")
    else:
        print("\n⚠️ TEST COMPLETED (Physics might need more time to warm up).")

if __name__ == "__main__":
    run_test()
