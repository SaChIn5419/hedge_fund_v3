import asyncio
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chimera_core.event_loop import ChimeraEventLoop

if __name__ == "__main__":
    print("==========================================")
    print("   CHIMERA INSTITUTIONAL ENGINE (WS)      ")
    print("==========================================")
    
    engine = ChimeraEventLoop()
    
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        print("\n🛑 Engine Stopped by User")
    except Exception as e:
        print(f"❌ Fatal Error: {e}")
