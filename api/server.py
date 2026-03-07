from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Optional

from ..websocket.manager import manager

from database.queries import (
    insert_sensor_data,
    insert_prediction
)

from services.data_service import (
    fetch_latest_vitals,
    fetch_day_timeline,
    fetch_week_timeline,
    fetch_month_timeline
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


# -----------------------------
# Pydantic model for validation
# -----------------------------
class SensorData(BaseModel):
    patient_id: int
    heart_rate: float
    resp_rate: float
    spo2: float
    temperature: float
    hrv: Optional[float] = None
    rrv: Optional[float] = None
    risk_score: float
    risk_level: str


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

    timestamp = datetime.now(timezone.utc)

    insert_sensor_data(
        data.patient_id,
        data.heart_rate,
        data.resp_rate,
        data.spo2,
        data.temperature,
        data.hrv,
        data.rrv,
        timestamp
    )

    insert_prediction(
        data.patient_id,
        data.risk_score,
        data.risk_level,
        timestamp
    )

    # Broadcast to dashboard clients
    await manager.broadcast({
        "patient_id": data.patient_id,
        "heart_rate": data.heart_rate,
        "resp_rate": data.resp_rate,
        "spo2": data.spo2,
        "temperature": data.temperature,
        "hrv": data.hrv,
        "rrv": data.rrv,
        "risk_score": data.risk_score,
        "risk_level": data.risk_level,
        "timestamp": str(timestamp)
    })

    return {"status": "data stored"}


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
        "heart_rate": data[0],
        "resp_rate": data[1],
        "spo2": data[2],
        "temperature": data[3],
        "hrv": data[4],
        "rrv": data[5],
        "timestamp": data[6]
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