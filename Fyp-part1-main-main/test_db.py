import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        database="mimic_demo",
        user="postgres",
        password="postgre.22"  # Try your password
    )
    print("Database connection successful!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")