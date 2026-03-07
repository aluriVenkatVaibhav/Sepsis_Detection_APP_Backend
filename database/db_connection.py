import psycopg2

DB_URL = "postgres://tsdbadmin:b5mvcl7c4bku41di@co5hse86mn.nurvahh53o.tsdb.cloud.timescale.com:36015/tsdb?sslmode=require"

def get_connection():
    return psycopg2.connect(DB_URL)