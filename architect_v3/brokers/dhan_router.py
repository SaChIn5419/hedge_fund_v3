from dhanhq import dhanhq
import sys

class DhanRouter:
    def __init__(self, client_id, access_token):
        self.dhan = dhanhq(client_id, access_token)
        status = self.dhan.get_fund_limits()
        if 'data' not in status:
            print("FATAL ERROR: Dhan API Connection Failed.")
            sys.exit(1)

    def get_live_capital(self):
        funds = self.dhan.get_fund_limits()
        return float(funds['data']['sodLimit'])

    def execute_basket(self, orders):
        print("\n--- ARCHITECT: EXECUTING LIVE NSE ORDERS ---")
        for order in orders:
            try:
                symbol = order['ticker'].replace(".NS", "")
                resp = self.dhan.place_order(
                    security_id=symbol,   
                    exchange_segment=self.dhan.NSE,
                    transaction_type=self.dhan.BUY,
                    quantity=order['quantity'],
                    order_type=self.dhan.MARKET,
                    product_type=self.dhan.CNC 
                )
                print(f"[SUCCESS] BOUGHT {order['quantity']} shares of {symbol}")
            except Exception as e:
                print(f"[FAILED] Ticker {order['ticker']} - Error: {e}")
