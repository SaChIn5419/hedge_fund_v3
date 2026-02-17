import time

# ---------------------------------------------
# STATES
# ---------------------------------------------
STATE_IDLE = "IDLE"
STATE_SIGNAL_READY = "SIGNAL_READY"
STATE_EXECUTING = "EXECUTING"
STATE_MONITORING = "MONITORING"
STATE_COOLDOWN = "COOLDOWN"

class ExecutionStateMachine:
    def __init__(self):
        self.current_state = STATE_IDLE
        self.last_execution_time = 0
        self.cooldown_period = 60 # Seconds
        self.active_orders = []

    def set_state(self, new_state):
        print(f"🔄 State Transition: {self.current_state} -> {new_state}")
        self.current_state = new_state

    def can_execute(self):
        """
        Checks if the system is allowed to enter EXECUTING state.
        """
        if self.current_state == STATE_COOLDOWN:
            elapsed = time.time() - self.last_execution_time
            if elapsed > self.cooldown_period:
                print("✅ Cooldown Expired. Returning to IDLE.")
                self.set_state(STATE_IDLE)
                return True
            else:
                return False
        
        return self.current_state == STATE_IDLE or self.current_state == STATE_SIGNAL_READY

    def update_state_post_execution(self, order_ids):
        """
        Transition after orders are sent.
        """
        self.active_orders = order_ids
        self.last_execution_time = time.time()
        self.set_state(STATE_MONITORING)
        
        # In a real system, we'd wait for fill confirmations.
        # Here we simulate a monitoring phase then go to cooldown.
        # For simplicity in Phase 1, we auto-transition to Cooldown.
        self.set_state(STATE_COOLDOWN)

    def log_state(self):
        return {
            "state": self.current_state,
            "last_exec": self.last_execution_time,
            "active_orders": len(self.active_orders)
        }
