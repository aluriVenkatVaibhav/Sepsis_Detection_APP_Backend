from collections import defaultdict, deque
import time

class PatientState:
    def __init__(self):
        self.mode = "IDLE"  # IDLE / TRAIN / MONITOR
        self.buffer = []    # store training windows
        self.baseline = None
        self.model = None
        self.last_updated = time.time()
        self.recent_packet_ids = deque(maxlen=512)
        self.recent_packet_set = set()

    def is_duplicate_packet(self, packet_id: str) -> bool:
        return packet_id in self.recent_packet_set

    def remember_packet(self, packet_id: str):
        if packet_id in self.recent_packet_set:
            return
        if len(self.recent_packet_ids) == self.recent_packet_ids.maxlen:
            oldest = self.recent_packet_ids.popleft()
            self.recent_packet_set.discard(oldest)
        self.recent_packet_ids.append(packet_id)
        self.recent_packet_set.add(packet_id)

# In-memory store (later can move to Redis/DB)
patient_states = defaultdict(PatientState)