from flask import Blueprint, request, jsonify, session
from config.database_config import get_db_connection
from psycopg2.extras import RealDictCursor
from functools import wraps

doctor_bp = Blueprint("doctor", __name__, url_prefix="/doctor")

# =========================
# DECORATOR (FIXED)
# =========================
def doctor_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session or session.get("role") != "doctor":
            return jsonify({"success": False, "error": "Unauthorized"}), 403
        return f(*args, **kwargs)
    return wrapper


# =========================
# GET MY PROFILE
# =========================
@doctor_bp.route("/profile")
@doctor_required
def profile():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
    SELECT             
        full_name,
        email,
        gender,
        specialization,
        department,
        COALESCE(experience_years, experience, 0) AS experience_years,
        status,
        phone_code,
        contact,
        date_of_birth,
        age,
        address
    FROM doctors
    WHERE user_id = %s
""", (session["user_id"],))

        data = cur.fetchone()


        if not data:
            return jsonify({
                "success": False,
                "error": "Doctor profile not found"
            }), 404
        

        cur.close()
        conn.close()

        return jsonify({"success": True, "data": data})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# UPDATE PROFILE
# =========================
@doctor_bp.route("/profile", methods=["PUT"])
@doctor_required
def update_profile():
    try:
        data = request.json
        user_id = session["user_id"]

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            UPDATE doctors
            SET 
                full_name = %s,
                email = %s,
                gender = %s,
                specialization = %s,
                department = %s,
                phone_code = %s,
                contact = %s,
                status = %s,
                experience_years = %s,
                date_of_birth = %s,
                age = %s,
                address = %s
            WHERE user_id = %s
        """, (
            data.get("full_name"),
            data.get("email"),
            data.get("gender"),
            data.get("specialization"),
            data.get("department"),
            data.get("phone_code"),
            data.get("contact"),
            data.get("status"),
            data.get("experience_years"),
            data.get("date_of_birth"),
            data.get("age"),
            data.get("address"),
            user_id
        ))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"success": True, "message": "Profile updated"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    

# =========================
# GET MY APPOINTMENTS
# =========================
@doctor_bp.route("/appointments")
@doctor_required
def get_appointments():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT
                a.id,
                a.patient_id,
                p.full_name AS patient,
                TO_CHAR(a.appointment_date, 'YYYY-MM-DD') AS date,
                TO_CHAR(a.appointment_time, 'HH24:MI') AS time,
                a.status
            FROM appointments a
            JOIN patient_profiles p ON a.patient_id = p.id
            JOIN doctors d ON a.doctor_id = d.id
            WHERE d.user_id = %s
            ORDER BY a.appointment_date DESC
        """, (session["user_id"],))

        rows = cur.fetchall()

        cur.close()
        conn.close()

        return jsonify({"success": True, "data": rows})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# CREATE APPOINTMENT
# =========================
@doctor_bp.route("/appointments", methods=["POST"])
@doctor_required
def create_appointment():
    try:
        data = request.json
        status = request.json.get("status")

        patient_id = data.get("patient_id")
        date = data.get("date")
        time = data.get("time")

        if not patient_id or not date or not time:
            return jsonify({"success": False, "error": "Missing fields"}), 400

        conn = get_db_connection()
        cur = conn.cursor()

        # get doctor id
        cur.execute("SELECT id FROM doctors WHERE user_id = %s", (session["user_id"],))
        doctor = cur.fetchone()

        if not doctor:
            return jsonify({"success": False, "error": "Doctor not found"}), 404

        doctor_id = doctor[0]

        cur.execute("""
            INSERT INTO appointments (patient_id, doctor_id, appointment_date, appointment_time, status)
            VALUES (%s, %s, %s, %s, %s)
""", (patient_id, doctor_id, date, time, status))

        conn.commit()

        cur.close()
        conn.close()

        return jsonify({"success": True, "message": "Appointment created"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# UPDATE APPOINTMENT STATUS
# =========================
@doctor_bp.route("/appointments/<int:id>", methods=["PUT"])
@doctor_required
def update_status(id):
    try:
        data = request.json
        status = request.json.get("status")

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT id FROM doctors WHERE user_id = %s", (session["user_id"],))
        doctor = cur.fetchone()

        if not doctor:
            return jsonify({"success": False, "error": "Doctor not found"}), 404

        doctor_id = doctor[0]

        cur.execute("""
            UPDATE appointments
            SET status = %s
            WHERE id = %s AND doctor_id = %s
""", (status, id, doctor_id))

        conn.commit()

        cur.close()
        conn.close()

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# DELETE APPOINTMENT
# =========================
@doctor_bp.route("/appointments/<int:id>", methods=["DELETE"])
@doctor_required
def delete_appointment(id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("DELETE FROM appointments WHERE id = %s", (id,))
        conn.commit()

        cur.close()
        conn.close()

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# =========================
# GET PATIENTS
# =========================
@doctor_bp.route("/patients")
@doctor_required
def patients():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT id, full_name AS name, age, gender
            FROM patient_profiles
            ORDER BY id DESC
        """)

        rows = cur.fetchall()

        cur.close()
        conn.close()

        return jsonify({"success": True, "data": rows})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# GET SINGLE PATIENT
# =========================
@doctor_bp.route("/patients/<int:id>", methods=["GET"])
@doctor_required
def get_patient(id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)


        cur.execute("""
            SELECT 
                p.id,
                p.full_name AS name,
                p.age,
                p.gender,

                ph.prediction,
                ph.risk_level,
                ph.temperature,
                ph.heart_rate,
                ph.respiratory_rate
                

            FROM patient_profiles p

            LEFT JOIN prediction_history ph 
                ON ph.user_id = p.user_id

            WHERE p.id = %s
            ORDER BY ph.id DESC
            LIMIT 1
        """, (id,))

        patient = cur.fetchone()

        if patient:
    # convert probability → percentage
            prob = patient.get("probability")

            if prob is not None:
                patient["risk_percentage"] = round(prob * 100, 2)
            else:
                patient["risk_percentage"] = 0

        cur.close()
        conn.close()

        if not patient:
            return jsonify({"success": False, "error": "Patient not found"}), 404

        return jsonify({"success": True, "data": patient})

    except Exception as e:
        print("PATIENT FETCH ERROR:", e)   # 👈 ADD THIS
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# MESSAGES
# =========================

@doctor_bp.route("/users", methods=["GET"])
@doctor_required
def get_users():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT 
                u.id,
                COALESCE(d.full_name, a.full_name, 'Unknown') AS full_name,
                u.role
            FROM users u
            LEFT JOIN doctors d ON d.user_id = u.id
            LEFT JOIN attendants a ON a.user_id = u.id
            WHERE u.role IN ('doctor', 'attendant')
            AND u.id != %s
        """, (session["user_id"],))

        users = cur.fetchall()

        cur.close()
        conn.close()

        return jsonify({"success": True, "data": users})

    except Exception as e:
        print("USERS ERROR:", e)
        return jsonify({"success": False, "error": str(e)}), 500
    
    
@doctor_bp.route("/messages", methods=["GET", "POST"])
@doctor_required
def messages():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        if request.method == "POST":
            d = request.json

            cur.execute("""
                INSERT INTO messages (sender_id, receiver_id, body)
                VALUES (%s, %s, %s)
            """, (
                session["user_id"],
                d["receiver_id"],
                d["body"]
            ))

            conn.commit()
            return jsonify({"success": True})

        cur.execute("""
    SELECT 
        m.id, 
        m.body, 
        m.sender_id, 
        m.receiver_id,

        -- sender name
        COALESCE(d1.full_name, a1.full_name, 'Unknown') AS sender_name,

        -- receiver name
        COALESCE(d2.full_name, a2.full_name, 'Unknown') AS receiver_name

    FROM messages m
    JOIN users u1 ON u1.id = m.sender_id
    LEFT JOIN doctors d1 ON d1.user_id = u1.id
    LEFT JOIN attendants a1 ON a1.user_id = u1.id

    JOIN users u2 ON u2.id = m.receiver_id
    LEFT JOIN doctors d2 ON d2.user_id = u2.id
    LEFT JOIN attendants a2 ON a2.user_id = u2.id

    WHERE m.sender_id = %s OR m.receiver_id = %s
    ORDER BY m.id DESC
""", (session["user_id"], session["user_id"]))

        rows = cur.fetchall()

        cur.close()
        conn.close()

        return jsonify({"success": True, "data": rows})

    except Exception as e:
     print("MESSAGES ERROR:", e)   # 👈 ADD THIS
     return jsonify({"success": False, "error": str(e)}), 500
    


@doctor_bp.route("/update-username", methods=["PUT"])
@doctor_required
def update_username():
    try:
        data = request.json
        new_username = data.get("username")

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            UPDATE users SET username = %s WHERE id = %s
        """, (new_username, session["user_id"]))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@doctor_bp.route("/update-email", methods=["PUT"])
@doctor_required
def update_email():
    try:
        data = request.json
        new_email = data.get("email")

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            UPDATE doctors SET email = %s WHERE user_id = %s
        """, (new_email, session["user_id"]))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
from werkzeug.security import generate_password_hash, check_password_hash

@doctor_bp.route("/change-password", methods=["PUT"])
@doctor_required
def change_password():
    try:
        data = request.get_json()
        print("REQUEST DATA:", data)

        if not data:
            return jsonify({"success": False, "error": "Invalid request"}), 400

        current = data.get("current_password")
        new = data.get("new_password")

        if not current or not new:
            return jsonify({"success": False, "error": "All fields required"}), 400

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("SELECT password_hash FROM users WHERE id = %s", (session["user_id"],))
        user = cur.fetchone()

        print("USER:", user)

        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404

        if not check_password_hash(user["password_hash"], current):
            return jsonify({"success": False, "error": "Incorrect current password"}), 400

        hashed = generate_password_hash(new)

        cur.execute("""
            UPDATE users SET password_hash = %s WHERE id = %s
        """, (hashed, session["user_id"]))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"success": True})

    except Exception as e:
        print("CHANGE PASSWORD ERROR:", e)
        return jsonify({"success": False, "error": str(e)}), 500
# =========================
# TEST ROUTE
# =========================
@doctor_bp.route("/", methods=["GET"])
def test_doctor():
    return jsonify({"message": "Doctor route working"})