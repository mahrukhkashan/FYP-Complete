import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        database="mimic_demo",
        user="postgres",
        password="postgres123",
        port=5432
    )
    print("Database connection successful!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
