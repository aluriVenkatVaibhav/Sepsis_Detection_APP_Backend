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

CREATE TABLE sensor_data (
    id SERIAL,
    patient_id INTEGER NOT NULL,

    heart_rate DOUBLE PRECISION,
    resp_rate DOUBLE PRECISION,
    temperature DOUBLE PRECISION,
    spo2 DOUBLE PRECISION,

    hrv DOUBLE PRECISION,
    rrv DOUBLE PRECISION,

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

CREATE TABLE predictions (
    patient_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,

    risk_score DOUBLE PRECISION,
    risk_level TEXT,

    PRIMARY KEY (patient_id, timestamp)
);