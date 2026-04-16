from datetime import datetime, timezone
from typing import Dict

from ml.sepsis_detector import SepsisDetector
from ml.models_factory import build_population_if, build_random_forest
from ml.vitals_types import VitalsSample

population_if = build_population_if()
rf_model = build_random_forest()

# ✅ KEEP DETECTORS
detectors: Dict[int, SepsisDetector] = {}

def get_detector(patient_id: int) -> SepsisDetector:
    if patient_id not in detectors:
        detectors[patient_id] = SepsisDetector(population_if, rf_model)
    return detectors[patient_id]


def process_vitals(data, baseline=None, personal_model=None):
    detector = get_detector(data.patient_id)

    # ✅ Inject baseline only once
    if baseline is not None and not detector.baseline_locked:
        detector.load_baseline(baseline, personal_model)

    sample = VitalsSample(
        timestamp=getattr(data, "timestamp", None) or datetime.now(timezone.utc),
        hr=data.hr,
        rr=data.rr,
        spo2=data.spo2,
        temp=data.temp,
        movement=data.movement,
        hrv=data.hrv or 0.0,
        rrv=data.rrv or 0.0
    )

    output = detector.process_monitoring_window(sample)

    return {
        "phase": "MONITORING",
        "status": output["status"],
        "final_score": output["final_score"],
        "sepsis_phase": output["sepsis_phase"],
        "baseline_confidence": output.get("baseline_confidence"),
        "baseline_state": output.get("baseline_state"),
        "full_output": output
    }