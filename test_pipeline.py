import random
from datetime import datetime, timezone
from sklearn.ensemble import IsolationForest

from ml.vitals_types import VitalsSample
from ml.baseline_establishment import BaselineEstablishment
from ml.sepsis_detector import SepsisDetector
from ml.models_factory import build_population_if, build_random_forest


# -----------------------------
# STEP 1: Generate fake vitals
# -----------------------------
def generate_sample(base_hr=75, noise=5):
    return VitalsSample(
        timestamp=datetime.now(timezone.utc),
        hr=base_hr + random.uniform(-noise, noise),
        rr=16 + random.uniform(-2, 2),
        spo2=98 + random.uniform(-1, 1),
        temp=36.8 + random.uniform(-0.3, 0.3),
        movement=random.uniform(0, 10),
        hrv=50 + random.uniform(-5, 5),
        rrv=20 + random.uniform(-2, 2)
    )


# -----------------------------
# STEP 2: TRAIN PHASE
# -----------------------------
print("\n=== TRAINING PHASE ===")

train_buffer = [generate_sample() for _ in range(5)]

baseline_builder = BaselineEstablishment()

baseline = None
for sample in train_buffer:
    result = baseline_builder.add_window(sample)
    if result is not None:
        baseline = result

if baseline is None:
    raise Exception("Baseline not created")

# Train personal Isolation Forest
X = [[
    s.hr, s.rr, s.spo2, s.temp,
    s.movement, s.hrv, s.rrv
] for s in train_buffer]

personal_if = IsolationForest()
personal_if.fit(X)

print("✅ Baseline + model trained")


# -----------------------------
# STEP 3: INIT DETECTOR
# -----------------------------
population_if = build_population_if()
rf_model = build_random_forest()

detector = SepsisDetector(population_if, rf_model)

# Inject baseline + model
detector.load_baseline(baseline, personal_if)

print("✅ Detector initialized")


# -----------------------------
# STEP 4: MONITORING PHASE
# -----------------------------
print("\n=== MONITORING PHASE ===")

for i in range(10):
    sample = generate_sample()

    output = detector.process_monitoring_window(sample)

    print(f"\nWindow {i+1}")
    print(f"Score: {output['final_score']}")
    print(f"Status: {output['status']}")
    print(f"Phase: {output['sepsis_phase']}")