import sys
import os
# Add project root to path (2 levels up)
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root)
# Also add Chimera_Pro_Live for secrets
sys.path.append(os.path.join(root, 'Chimera_Pro_Live'))

try:
    from chimera_execution.secrets import TOTP_SECRET
except ImportError:
    # Fallback
    try:
        from secrets import TOTP_SECRET
    except:
        print("ERROR: Could not load secrets.")
        sys.exit(1)
        
import pyotp

print(f"Original Secret: '{TOTP_SECRET}'")
print(f"Length: {len(TOTP_SECRET)}")

# Try Clean
cleaned = TOTP_SECRET.replace(" ", "").strip()
print(f"Cleaned Secret: '{cleaned}'")

totp = pyotp.TOTP(cleaned)
print(f"Generated Code: {totp.now()}")
print("SUCCESS: Cleaning works.")


# End of script
