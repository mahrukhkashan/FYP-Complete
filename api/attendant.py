from flask import Blueprint, request, jsonify, session
from config.database_config import get_db_connection
from psycopg2.extras import RealDictCursor
from werkzeug.security import check_password_hash, generate_password_hash

attendant_bp = Blueprint("attendant", __name__, url_prefix="/attendant")


# =========================
# AUTH CHECK
# =========================
def attendant_required():
    return "user_id" in session and session.get("role") == "attendant"


# =========================
# GET PATIENTS (DASHBOARD)
# =========================
@attendant_bp.route("/patients", methods=["GET"])
def get_patients():
    if not attendant_required():
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT id,
               full_name as name,
               room_number as room,
               status,
               priority,
               flow
        FROM patient_profiles
    """)

    return jsonify({"patients": cursor.fetchall()})


# =========================
# UPDATE PATIENT STATUS
# =========================
@attendant_bp.route("/update-status", methods=["POST"])
def update_status():
    if not attendant_required():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE patient_profiles
SET 
    room_number = 'General Ward',
    status = 'Awaiting',
    priority = 'Normal',
    flow = 'admission'
WHERE room_number IS NULL;
        WHERE id=%s
    """, (data["status"], data["patient_id"]))

    conn.commit()
    return jsonify({"message": "Status updated"})

# =========================
# PATIENT FLOW
# =========================
@attendant_bp.route("/patient-flow", methods=["GET"])
def patient_flow():
    if not attendant_required():
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT id, full_name as name,
               room_number as room,
               priority, flow
        FROM patient_profiles
    """)

    return jsonify({"flow": cursor.fetchall()})

# =========================
# TASKS
# =========================
@attendant_bp.route("/tasks", methods=["GET"])
def get_tasks():
    if not attendant_required():
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
    SELECT t.id, t.time, p.full_name as patient,
           t.task, t.priority, t.status,
           COALESCE(a.full_name, 'Unassigned') as assignedTo,
           CASE 
               WHEN t.assigned_to = %s THEN true
               ELSE false
           END as isMine
    FROM tasks t
    JOIN patient_profiles p ON t.patient_id = p.id
    LEFT JOIN attendants a ON t.assigned_to::int = a.user_id
""", (session["user_id"],))
    return jsonify({"tasks": cursor.fetchall()})

# =========================
# ACCEPT TASK
# =========================
@attendant_bp.route("/accept-task", methods=["POST"])
def accept_task():
    if not attendant_required():
        return jsonify({"error": "Unauthorized"}), 401

    tid = request.json["task_id"]

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE tasks
        SET assigned_to=%s, status='In-Progress'
        WHERE id=%s
    """, (session["user_id"], tid))

    conn.commit()
    return jsonify({"message": "Task accepted"})

# =========================
# COMPLETE TASK
# =========================
@attendant_bp.route("/complete-task", methods=["POST"])
def complete_task():
    if not attendant_required():
        return jsonify({"error": "Unauthorized"}), 401

    tid = request.json["task_id"]

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM tasks WHERE id=%s", (tid,))
    conn.commit()

    return jsonify({"message": "Task completed"})

# =========================
# PROFILE
# =========================
@attendant_bp.route("/profile", methods=["GET"])
def get_profile():
    if not attendant_required():
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT *
        FROM attendants
        WHERE user_id=%s
    """, (session["user_id"],))

    return jsonify({"profile": cursor.fetchone()})

@attendant_bp.route("/profile", methods=["POST"])
def update_profile():
    if not attendant_required():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE attendants
        SET full_name=%s,
            email=%s,
            gender=%s,
            dob=%s,
            contact=%s,
            address=%s,
            shift=%s,
            notes=%s
        WHERE user_id=%s
    """, (
        data["name"], data["email"], data["gender"],
        data["dob"], data["contact"],
        data["address"], data["shift"],
        data["notes"], session["user_id"]
    ))

    conn.commit()
    return jsonify({"message": "Profile updated"})

# =========================
# SETTINGS
# =========================
@attendant_bp.route("/change-password", methods=["POST"])
def change_password():
    if not attendant_required():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT password FROM users WHERE id=%s", (session["user_id"],))
    user = cursor.fetchone()

    if not check_password_hash(user[0], data["current"]):
        return jsonify({"error": "Wrong password"}), 400

    new_hash = generate_password_hash(data["new"])

    cursor.execute("""
        UPDATE users SET password=%s WHERE id=%s
    """, (new_hash, session["user_id"]))

    conn.commit()
    return jsonify({"message": "Password updated"})


@attendant_bp.route("/add-task", methods=["POST"])
def add_task():
    if not attendant_required():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO tasks (patient_id, task, priority, time, status, assigned_to)
        VALUES (%s, %s, %s, %s, 'Pending', %s)
    """, (
        data["patient_id"],
        data["task"],
        data["priority"],
        data["time"],
        session["user_id"]   # 🔥 THIS IS THE KEY FIX
    ))

    conn.commit()
    conn.close()

    return jsonify({"message": "Task added"})

# =========================
# LOGOUT
# =========================
@attendant_bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})