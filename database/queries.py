from .db_connection import get_connection


def insert_patient(name, age, gender):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO patients (name, age, gender)
        VALUES (%s,%s,%s)
        RETURNING patient_id
        """,
        (name, age, gender)
    )

    patient_id = cur.fetchone()[0]

    conn.commit()
    cur.close()
    conn.close()

    return patient_id


def insert_sensor_data(patient_id, heart_rate, resp_rate, spo2, temperature, hrv, rrv, timestamp):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO sensor_data
        (patient_id, heart_rate, resp_rate, spo2, temperature, hrv, rrv, timestamp)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (patient_id, heart_rate, resp_rate, spo2, temperature, hrv, rrv, timestamp)
    )

    conn.commit()
    cur.close()
    conn.close()


def insert_prediction(patient_id, risk_score, risk_level, timestamp):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO predictions
        (patient_id, risk_score, risk_level, timestamp)
        VALUES (%s,%s,%s,%s)
        """,
        (patient_id, risk_score, risk_level, timestamp)
    )

    conn.commit()
    cur.close()
    conn.close()


def get_latest_vitals(patient_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT heart_rate, resp_rate, spo2, temperature, hrv, rrv, timestamp
        FROM sensor_data
        WHERE patient_id = %s
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (patient_id,)
    )

    result = cur.fetchone()

    cur.close()
    conn.close()

    return result


def get_day_timeline(patient_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            time_bucket('1 minute', timestamp) AS bucket,
            AVG(heart_rate),
            AVG(spo2),
            AVG(temperature),
            AVG(resp_rate),
            AVG(hrv),
            AVG(rrv)
        FROM sensor_data
        WHERE patient_id = %s
        AND timestamp > NOW() - INTERVAL '1 day'
        GROUP BY bucket
        ORDER BY bucket
        """,
        (patient_id,)
    )

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results


def get_week_timeline(patient_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            time_bucket('10 minutes', timestamp) AS bucket,
            AVG(heart_rate),
            AVG(spo2),
            AVG(temperature),
            AVG(resp_rate),
            AVG(hrv),
            AVG(rrv)
        FROM sensor_data
        WHERE patient_id = %s
        AND timestamp > NOW() - INTERVAL '7 days'
        GROUP BY bucket
        ORDER BY bucket
        """,
        (patient_id,)
    )

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results


def get_month_timeline(patient_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            time_bucket('1 hour', timestamp) AS bucket,
            AVG(heart_rate),
            AVG(spo2),
            AVG(temperature),
            AVG(resp_rate),
            AVG(hrv),
            AVG(rrv)
        FROM sensor_data
        WHERE patient_id = %s
        AND timestamp > NOW() - INTERVAL '30 days'
        GROUP BY bucket
        ORDER BY bucket
        """,
        (patient_id,)
    )

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results