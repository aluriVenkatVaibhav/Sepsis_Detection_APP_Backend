-- Enable Timescale extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

--------------------------------------------------
-- Patients table
--------------------------------------------------

CREATE TABLE IF NOT EXISTS patients (
    patient_id SERIAL PRIMARY KEY,
    name TEXT,
    age INT,
    gender TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

--------------------------------------------------
-- Sensor data table (time-series)
--------------------------------------------------

CREATE TABLE IF NOT EXISTS sensor_data (
    id SERIAL,
    patient_id INT REFERENCES patients(patient_id),

    heart_rate FLOAT,
    resp_rate FLOAT,
    spo2 FLOAT,
    temperature FLOAT,
    hrv FLOAT,
    rrv FLOAT,

    timestamp TIMESTAMPTZ NOT NULL,

    PRIMARY KEY (id, timestamp)
);

--------------------------------------------------
-- Convert to Timescale hypertable
--------------------------------------------------

SELECT create_hypertable('sensor_data', 'timestamp', if_not_exists => TRUE);

--------------------------------------------------
-- Predictions table
--------------------------------------------------

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(patient_id),

    risk_score FLOAT,
    risk_level TEXT,

    timestamp TIMESTAMPTZ DEFAULT NOW()
);