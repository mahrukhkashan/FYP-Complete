from flask import Blueprint, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from config.database_config import get_db_connection
from psycopg2.extras import RealDictCursor

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


# =========================
# HELPER (GET DATA SAFELY)
# =========================
def get_request_data():
    if request.is_json:
        return request.get_json()
    return request.form


# =========================
# REGISTER
# =========================
@auth_bp.route("/register", methods=["POST"])
def register():
    data = get_request_data()

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role", "patient")
    full_name = data.get("full_name", "")

    if not username or not password or not email:
        return jsonify({"success": False, "error": "Missing required fields"}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if user exists
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            return jsonify({"success": False, "error": "User already exists"}), 409

        password_hash = generate_password_hash(password)

        cur.execute("""
            INSERT INTO users (username, email, password_hash, role, full_name)
            VALUES (%s, %s, %s, %s, %s)
        """, (username, email, password_hash, role, full_name))

        conn.commit()

        cur.close()
        conn.close()

        return jsonify({"success": True, "message": "User registered successfully"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# LOGIN
# =========================
@auth_bp.route("/login", methods=["POST"])
def login():
    data = get_request_data()

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"success": False, "error": "Missing credentials"}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT id, username, password_hash, role, full_name
            FROM users
            WHERE username = %s
        """, (username,))

        user = cur.fetchone()

        cur.close()
        conn.close()

        if not user or not check_password_hash(user["password_hash"], password):
            return jsonify({"success": False, "error": "Invalid credentials"}), 401

        # 🔥 CRITICAL FIX — CLEAR OLD SESSION
        session.clear()

        # ✅ SET SESSION PROPERLY
        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["role"] = user["role"]
        session["full_name"] = user["full_name"]

        return jsonify({
            "success": True,
            "message": "Login successful",
            "role": user["role"]
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# LOGOUT
# =========================
@auth_bp.route("/logout", methods=["POST", "GET"])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out"})


# =========================
# DEBUG SESSION (REMOVE LATER)
# =========================
@auth_bp.route("/debug", methods=["GET"])
def debug():
    return jsonify(dict(session))