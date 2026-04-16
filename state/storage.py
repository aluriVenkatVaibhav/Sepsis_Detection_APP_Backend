import joblib
import os
import json
from datetime import datetime

from ml.vitals_types import BaselineData

BASE_DIR = "patient_data"

def get_patient_dir(patient_id):
    path = os.path.join(BASE_DIR, str(patient_id))
    os.makedirs(path, exist_ok=True)
    return path

def save_baseline(patient_id, baseline):
    path = os.path.join(get_patient_dir(patient_id), "baseline.json")
    with open(path, "w") as f:
        json.dump(_baseline_to_dict(baseline), f)

def load_baseline(patient_id):
    path = os.path.join(get_patient_dir(patient_id), "baseline.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return _baseline_from_dict(json.load(f))


def _baseline_to_dict(baseline: BaselineData) -> dict:
    return {
        "mode": baseline.mode,
        "confidence": baseline.confidence,
        "confidence_breakdown": baseline.confidence_breakdown,
        "baseline_means": baseline.baseline_means,
        "baseline_stds": baseline.baseline_stds,
        "locked_at": baseline.locked_at.isoformat(),
    }


def _baseline_from_dict(d: dict) -> BaselineData:
    return BaselineData(
        mode=d["mode"],
        confidence=float(d["confidence"]),
        confidence_breakdown=d["confidence_breakdown"],
        baseline_means={k: float(v) for k, v in d["baseline_means"].items()},
        baseline_stds={k: float(v) for k, v in d["baseline_stds"].items()},
        locked_at=datetime.fromisoformat(d["locked_at"]) if "locked_at" in d and d["locked_at"] else datetime.now(),
    )

def save_model(patient_id, model):
    path = os.path.join(get_patient_dir(patient_id), "model.pkl")
    joblib.dump(model, path)

def load_model(patient_id):
    path = os.path.join(get_patient_dir(patient_id), "model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)