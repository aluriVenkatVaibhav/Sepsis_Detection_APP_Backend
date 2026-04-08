from pydantic import BaseModel
from typing import List
from datetime import datetime


class SensorData(BaseModel):
    patient_id: int
    hr: float
    temp: float
    rr: float
    spo2: float
    hrv: float
    rrv: float
    movement: float
    timestamp: datetime


class PredictionData(BaseModel):
    patient_id: int
    predicted_sepsis: bool
    current_risk_score: float
    risk_scores: List[float]
    score_timestamps: List[datetime]