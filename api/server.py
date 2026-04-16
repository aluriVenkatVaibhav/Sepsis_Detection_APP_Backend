from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Optional
from services.ml_service import process_vitals

from websocket.manager import manager

from state.patient_state import patient_states
from state.storage import load_baseline, load_model
from routes.train import router as train_router
from ml.vitals_types import VitalsSample

from database.queries import (
    insert_sensor_data,
    insert_prediction
)

from services.data_service import (
    fetch_latest_vitals,
    fetch_day_timeline,
    fetch_week_timeline,
    fetch_month_timeline,
    fetch_day_prediction_timeline,
    fetch_week_prediction_timeline,
    fetch_month_prediction_timeline,
)

app = FastAPI()


# -----------------------------
# CORS (allow mobile app access)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train_router)

# -----------------------------
# Pydantic model for validation
# -----------------------------
class SensorData(BaseModel):
    patient_id: int
    hr: float
    temp: float
    rr: float
    spo2: float
    hrv: Optional[float] = None
    rrv: Optional[float] = None
    movement: float
    # Timestamp sent by the client (used for derivative timing / window alignment)
    timestamp: Optional[datetime] = None
    packet_seq: Optional[int] = None


# -----------------------------
# Health check endpoint
# -----------------------------
@app.get("/")
def root():
    return {"status": "SepsisGuard backend running"}


# -----------------------------
# Sensor data ingestion
# -----------------------------
@app.post("/sensor-data")
async def receive_sensor_data(data: SensorData):

    timestamp = data.timestamp or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    state = patient_states[data.patient_id]
    packet_key = (
        f"{data.patient_id}:"
        f"{data.packet_seq if data.packet_seq is not None else timestamp.isoformat()}:"
        f"{round(data.hr, 3)}:{round(data.rr, 3)}:{round(data.spo2, 3)}:{round(data.temp, 3)}"
    )
    if state.is_duplicate_packet(packet_key):
        return {
            "message": "duplicate packet ignored",
            "status": "DUPLICATE_IGNORED",
        }
    state.remember_packet(packet_key)

    insert_sensor_data(
        data.patient_id,
        data.hr,
        data.temp,
        data.rr,
        data.spo2,
        data.hrv,
        data.rrv,
        data.movement,
        timestamp
    )

    # -----------------------------
    # Run ML model
    # -----------------------------
    state.last_updated = datetime.now(timezone.utc).timestamp()

    # -----------------------------------
    # TRAIN MODE
    # -----------------------------------
    if state.mode == "TRAIN":
        

        sample = VitalsSample(
                timestamp=timestamp,
                hr=data.hr,
                rr=data.rr,
                spo2=data.spo2,
                temp=data.temp,
                movement=data.movement,
                hrv=data.hrv or 0.0,
                rrv=data.rrv or 0.0
            )

        state.buffer.append(sample)

        return {
            "message": "collecting training data",
            "status": "TRAINING",
            "windows_collected": len(state.buffer)
        }

    # -----------------------------------
    # LOAD BASELINE + MODEL (once)
    # -----------------------------------
    if state.baseline is None:
        state.baseline = load_baseline(data.patient_id)

    if state.model is None:
        state.model = load_model(data.patient_id)

    # -----------------------------------
    # NO BASELINE → BLOCK MONITORING
    # -----------------------------------
    if state.baseline is None:
        return {
            "error": "No baseline found. Please press TRAIN first."
        }

    if state.model is None:
        return {
            "error": "No personal model found. Please press TRAIN first."
        }

    # -----------------------------------
    # MONITOR MODE
    # -----------------------------------
    ml_result = process_vitals(
        data,
        baseline=state.baseline,
        personal_model=state.model
    )

    # -----------------------------
    # Store prediction ONLY if monitoring phase
    # -----------------------------
    if ml_result["phase"] == "MONITORING":
        insert_prediction(
            data.patient_id,
            ml_result["status"] in ["HIGH_RISK", "CRITICAL"],
            ml_result["final_score"],
            [ml_result["final_score"]],
            [timestamp]
        )
    

    # Broadcast to dashboard clients
    await manager.broadcast({
        "patient_id": data.patient_id,
        "hr": data.hr,
        "rr": data.rr,
        "spo2": data.spo2,
        "temp": data.temp,
        "hrv": data.hrv or 0.0,
        "rrv": data.rrv or 0.0,
        "movement": data.movement,
        "timestamp": str(timestamp),
        "ml": ml_result
    })

    return {
        "message": "data processed",
        "ml": ml_result
    }


# -----------------------------
# WebSocket endpoint
# -----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# -----------------------------
# Latest vitals endpoint
# -----------------------------
@app.get("/latest-vitals/{patient_id}")
def latest_vitals(patient_id: int):

    data = fetch_latest_vitals(patient_id)

    if not data:
        return {"error": "No data found"}

    return {
        "hr": data[0],
        "rr": data[1],
        "spo2": data[2],
        "temp": data[3],
        "hrv": data[4],
        "rrv": data[5],
        "movement": data[6],
        "timestamp": data[7]
    }


# -----------------------------
# Timeline analytics endpoints
# -----------------------------
@app.get("/timeline/day/{patient_id}")
def day_timeline(patient_id: int):

    data = fetch_day_timeline(patient_id)
    return {"data": data}


@app.get("/timeline/week/{patient_id}")
def week_timeline(patient_id: int):

    data = fetch_week_timeline(patient_id)
    return {"data": data}


@app.get("/timeline/month/{patient_id}")
def month_timeline(patient_id: int):

    data = fetch_month_timeline(patient_id)
    return {"data": data}


@app.get("/prediction-timeline/day/{patient_id}")
def day_prediction_timeline(patient_id: int):
    data = fetch_day_prediction_timeline(patient_id)
    return {"data": data}


@app.get("/prediction-timeline/week/{patient_id}")
def week_prediction_timeline(patient_id: int):
    data = fetch_week_prediction_timeline(patient_id)
    return {"data": data}


@app.get("/prediction-timeline/month/{patient_id}")
def month_prediction_timeline(patient_id: int):
    data = fetch_month_prediction_timeline(patient_id)
    return {"data": data}