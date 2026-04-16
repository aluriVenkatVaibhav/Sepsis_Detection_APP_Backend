from collections import defaultdict
import time

class PatientState:
    def __init__(self):
        self.mode = "IDLE"  # IDLE / TRAIN / MONITOR
        self.buffer = []    # store training windows
        self.baseline = None
        self.model = None
        self.last_updated = time.time()

# In-memory store (later can move to Redis/DB)
patient_states = defaultdict(PatientState)