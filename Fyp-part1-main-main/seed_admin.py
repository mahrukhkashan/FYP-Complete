# create_admin.py
from werkzeug.security import generate_password_hash
from config.database_config import get_db_connection

def create_admin():
    conn = get_db_connection()
    cur = conn.cursor()

    # Check if admin already exists
    cur.execute("SELECT id FROM users WHERE username = %s", ("admin",))
    if cur.fetchone():
        print("Admin user already exists!")
    else:
        # Insert admin user
        cur.execute("""
            INSERT INTO users (username, email, password_hash, role, full_name)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            "admin",
            "admin@example.com",
            generate_password_hash("admin123"),  # hashed password
            "admin",
            "Admin User"
        ))
        conn.commit()
        print("Admin user created successfully!")

    cur.close()
    conn.close()

if __name__ == "__main__":
    create_admin()
