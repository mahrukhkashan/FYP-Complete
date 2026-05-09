from flask import Blueprint, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from config.database_config import get_db_connection

auth_bp = Blueprint("auth", __name__)

# ---------------- REGISTER ----------------
@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.form  # ✅ FIX

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role", "patient")
    full_name = data.get("full_name", "")

    if not username or not password or not email:
        return "Missing required fields", 400

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", ("admin",))
    if not cur.fetchone():
        cur.execute("""
         INSERT INTO users (username, email, password_hash, role, full_name)
         VALUES (%s, %s, %s, %s, %s)
    """, ("admin", "admin@example.com", generate_password_hash("admin123"), "admin", "Admin User"))
    conn.commit()

    # Check if user exists
    cur.execute("SELECT id FROM users WHERE username=%s", (username,))
    if cur.fetchone():
        return "User already exists", 409

    password_hash = generate_password_hash(password)

    cur.execute("""
        INSERT INTO users (username, email, password_hash, role, full_name)
        VALUES (%s, %s, %s, %s, %s)
    """, (username, email, password_hash, role, full_name))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("login_page"))


# ---------------- LOGIN ----------------
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.form  # ✅ FIX

    username = data.get("username")
    password = data.get("password")

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, password_hash, role, full_name
        FROM users
        WHERE username=%s
    """, (username,))

    user = cur.fetchone()
    cur.close()
    conn.close()

    if not user or not check_password_hash(user[1], password):
        return "Invalid credentials", 401

    # Session
    session["user_id"] = user[0]
    session["username"] = username
    session["role"] = user[2]
    session["full_name"] = user[3]

    return redirect(url_for("dashboard"))


# ---------------- LOGOUT ----------------
@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))
