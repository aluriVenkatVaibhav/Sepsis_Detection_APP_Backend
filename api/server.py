from fastapi import FastAPI
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from backend.websocket.manager import manager

from backend.database.queries import (
    insert_sensor_data,
    insert_prediction
)

from backend.services.data_service import (
    fetch_latest_vitals,
    fetch_day_timeline,
    fetch_week_timeline,
    fetch_month_timeline
)

app = FastAPI()


@app.get("/")
def root():
    return {"status": "SepsisGuard backend running"}


@app.post("/sensor-data")
async def receive_sensor_data(data: dict):

    patient_id = data["patient_id"]

    heart_rate = data["heart_rate"]
    resp_rate = data["resp_rate"]
    spo2 = data["spo2"]
    temperature = data["temperature"]
    hrv = data.get("hrv", None)
    rrv = data.get("rrv", None)

    risk_score = data["risk_score"]
    risk_level = data["risk_level"]

    timestamp = datetime.utcnow()

    insert_sensor_data(
        patient_id,
        heart_rate,
        resp_rate,
        spo2,
        temperature,
        hrv,
        rrv,
        timestamp
    )

    insert_prediction(
        patient_id,
        risk_score,
        risk_level,
        timestamp
    )
    
    await manager.broadcast({
        "patient_id": patient_id,
        "heart_rate": heart_rate,
        "resp_rate": resp_rate,
        "spo2": spo2,
        "temperature": temperature,
        "hrv": hrv,
        "rrv": rrv,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "timestamp": str(timestamp)
    })

    return {"status": "data stored"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/latest-vitals/{patient_id}")
def latest_vitals(patient_id: int):

    data = fetch_latest_vitals(patient_id)

    return {
        "heart_rate": data[0],
        "resp_rate": data[1],
        "spo2": data[2],
        "temperature": data[3],
        "hrv": data[4],
        "rrv": data[5],
        "timestamp": data[6]
    }
    
    
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