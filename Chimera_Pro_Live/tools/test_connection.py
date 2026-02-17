import sys
import os
import pyotp
import base64
import binascii
from SmartApi import SmartConnect

# Add parent directory to path to locate secrets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from chimera_execution.secrets import API_KEY, CLIENT_ID, PASSWORD, TOTP_SECRET
    print("✅ Secrets loaded successfully.")
except ImportError:
    print("❌ Error: chimera_execution/secrets.py not found or missing variables.")
    sys.exit(1)

def fix_totp_secret(secret):
    """
    Attempts to fix common TOTP secret format issues (spaces, hex vs base32).
    """
    secret = secret.replace(" ", "").upper()
    
    # Check if it works as-is (Base32)
    try:
        pyotp.TOTP(secret).now()
        return secret
    except:
        pass
        
    # Try interpreting as Hex and converting to Base32
    try:
        # Pad if odd length (hex strings should be even, but just in case)
        if len(secret) % 2 != 0: secret = "0" + secret
            
        bytes_secret = binascii.unhexlify(secret)
        base32_secret = base64.b32encode(bytes_secret).decode('utf-8').replace('=', '')
        
        # Verify if this new secret works
        pyotp.TOTP(base32_secret).now()
        print(f"🔹 Auto-converted Hex Secret to Base32: {base32_secret}")
        return base32_secret
    except Exception as e:
        print(f"⚠️ Could not auto-convert secret: {e}")
        
    return secret

def test_login():
    print(f"🔐 Testing Credentials for Client ID: {CLIENT_ID}")
    
    clean_secret = fix_totp_secret(TOTP_SECRET)
    
    # 1. Initialize API Object
    try:
        api = SmartConnect(api_key=API_KEY)
        print("🔹 SmartConnect Object Initialized.")
    except Exception as e:
        print(f"❌ Failed to init SmartConnect: {e}")
        return

    # 2. Generate TOTP
    try:
        otp = pyotp.TOTP(clean_secret).now()
        print(f"🔹 Generated TOTP: {otp}")
    except Exception as e:
        print(f"❌ Failed to generate TOTP: {e}")
        print("💡 Hint: Ensure your TOTP Secret is a valid Base32 string (A-Z, 2-7).")
        return

    # 3. Login
    try:
        data = api.generateSession(CLIENT_ID, PASSWORD, otp)
        
        if data['status']:
            print("\n🚀 LOGIN SUCCESSFUL!")
            print(f"AUTH TOKEN: {data['data']['jwtToken'][:10]}... (Hidden)")
            
            # 4. Fetch Profile (Test Request)
            try:
                profile = api.getProfile(api.refreshToken)
                print(f"\n👤 User Profile: {profile['data']['name']}")
                print(f"📧 Email: {profile['data']['email']}")
            except:
                print("⚠️ Could not fetch profile (Minor issue).")
                
            # 5. Fetch Funds
            try:
                # rms = api.rmsLimit() 
                # print(f"💰 Funds Available: {rms['data']['net']}")
                pass
            except:
                pass
                
        else:
            print(f"\n❌ LOGIN FAILED: {data['message']}")
            
    except Exception as e:
        print(f"\n❌ CRITICAL LOGIN ERROR: {e}")

if __name__ == "__main__":
    test_login()
