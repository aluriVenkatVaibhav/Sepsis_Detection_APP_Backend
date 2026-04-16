from database.queries import (
    get_latest_vitals,
    get_day_timeline,
    get_week_timeline,
    get_month_timeline,
    get_day_prediction_timeline,
    get_week_prediction_timeline,
    get_month_prediction_timeline,
)

def fetch_latest_vitals(patient_id):
    return get_latest_vitals(patient_id)


def fetch_day_timeline(patient_id):
    return get_day_timeline(patient_id)


def fetch_week_timeline(patient_id):
    return get_week_timeline(patient_id)


def fetch_month_timeline(patient_id):
    return get_month_timeline(patient_id)


def fetch_day_prediction_timeline(patient_id):
    return get_day_prediction_timeline(patient_id)


def fetch_week_prediction_timeline(patient_id):
    return get_week_prediction_timeline(patient_id)


def fetch_month_prediction_timeline(patient_id):
    return get_month_prediction_timeline(patient_id)