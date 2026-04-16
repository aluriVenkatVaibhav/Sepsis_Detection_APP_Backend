from fastapi import APIRouter
from state.patient_state import patient_states

router = APIRouter()

@router.post("/train-start")
def start_training(patient_id: int):
    state = patient_states[patient_id]
    state.mode = "TRAIN"
    state.buffer = []
    return {"status": "TRAINING_STARTED"}

@router.post("/train-stop")
def stop_training(patient_id: int):
    state = patient_states[patient_id]

    if len(state.buffer) < 5:
        return {"error": "Not enough data for training"}

    # Build patient baseline (Phase A: 5 windows) + personal Isolation Forest
    from ml.baseline_establishment import BaselineEstablishment

    baseline_builder = BaselineEstablishment()
    baseline = None
    for sample in state.buffer:
        result = baseline_builder.add_window(sample)
        if result is not None:
            baseline = result

    if baseline is None or baseline_builder.personal_if is None:
        return {"error": "Baseline build failed. Try training again."}

    model = baseline_builder.personal_if

    # Save
    from state.storage import save_baseline, save_model
    save_baseline(patient_id, baseline)
    save_model(patient_id, model)

    # Update state
    state.baseline = baseline
    state.model = model
    state.mode = "MONITOR"

    # ✅ ADD THIS LINE HERE
    state.buffer = []   # clear training data
    
    return {"status": "TRAINING_COMPLETED"}