from flask import Blueprint, request, jsonify, session
import psycopg2
import psycopg2.extras
from config.database_config import get_db_connection
from werkzeug.security import generate_password_hash

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


# =========================
# AUTH CHECK
# =========================
def admin_required():
    if "user_id" not in session or session.get("role") != "admin":
        return False
    return True


# =========================
# DASHBOARD STATS
# =========================
@admin_bp.route("/stats")
def stats():
    if not admin_required():
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM patients")
    patients = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM appointments")
    appointments = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM doctors WHERE status='Available'")
    doctors = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM attendants")
    attendants = cur.fetchone()[0]

    cur.close()
    conn.close()

    return jsonify({
        "patients": patients,
        "appointments": appointments,
        "doctors": doctors,
        "attendants": attendants
    })
from psycopg2.extras import RealDictCursor
import bcrypt

def required_field(val, name):
    if val is None or str(val).strip() == "":
        raise ValueError(f"{name} is required")
    return val

from flask import request, jsonify, session
from psycopg2.extras import RealDictCursor
import bcrypt


def required_field(val, name):
    if val is None or str(val).strip() == "":
        raise ValueError(f"{name} is required")
    return val


@admin_bp.route("/patients", methods=["GET", "POST"])
def patients():
    if "user_id" not in session or session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # =========================
    # CREATE PATIENT
    # =========================
    if request.method == "POST":
        d = request.json

        try:
            # =========================
            # REQUIRED FIELD CHECKS
            # =========================
            required_field(d.get("name"), "Full name")
            required_field(d.get("username"), "Username")
            required_field(d.get("password"), "Password")
            required_field(d.get("gender"), "Gender")
            required_field(d.get("dob"), "Date of birth")
            required_field(d.get("age"), "Age")
            required_field(d.get("blood"), "Blood group")
            required_field(d.get("code"), "Country code")
            required_field(d.get("contact"), "Contact number")
            required_field(d.get("admType"), "Admission type")
            required_field(d.get("diagnosis"), "Diagnosis")
            required_field(d.get("address"), "Address")

            cur.execute("SELECT 1 FROM users WHERE username = %s", (d["username"],))
            if cur.fetchone():
             return jsonify({"error": "Username already exists"}), 409
            # =========================
            # CREATE USER
            # =========================
            hashed = generate_password_hash(d["password"])


            cur.execute("""
                INSERT INTO users (username, role, password_hash)
                VALUES (%s, 'patient', %s)
                RETURNING id
            """, (d["username"].strip(), hashed))

            user_id = cur.fetchone()["id"]

            # =========================
            # CREATE PATIENT PROFILE
            # =========================
            cur.execute("""
                INSERT INTO patient_profiles (
                    user_id,
                    full_name,
                    gender,
                    date_of_birth,
                    age,
                    blood_group,
                    contact_country_code,
                    contact_number,
                    admission_type,
                    admission_date,
                    diagnosis,
                    address
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                user_id,
                d["name"].strip(),
                d["gender"],
                d["dob"],
                int(d["age"]),
                d["blood"],
                d["code"],
                d["contact"].strip(),
                d["admType"],
                d.get("admDate"),   # ✅ OPTIONAL
                d["diagnosis"],
                d["address"]
            ))

            conn.commit()
            return jsonify({"success": True})

        except ValueError as ve:
            conn.rollback()
            return jsonify({"error": str(ve)}), 400

        except Exception as e:
            conn.rollback()
            print("PATIENT SAVE ERROR:", e)
            return jsonify({"error": "Database error"}), 500

        finally:
            cur.close()
            conn.close()

    # =========================
    # LIST PATIENTS
    # =========================
    cur.execute("""
    SELECT
        p.id,
        p.full_name        AS name,
        p.gender,
        p.age,
        p.blood_group     AS blood,
        p.admission_type  AS type,
        p.diagnosis,
        p.address,
        p.contact_country_code AS code,
        p.contact_number  AS contact
    FROM patient_profiles p
    ORDER BY p.id DESC
""")


    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(rows)


@admin_bp.route("/patients/<int:id>", methods=["DELETE"])
def delete_patient(id):
    if "user_id" not in session or session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # get linked user_id
        cur.execute(
            "SELECT user_id FROM patient_profiles WHERE id = %s",
            (id,)
        )
        row = cur.fetchone()

        if not row:
            return jsonify({"error": "Patient not found"}), 404

        user_id = row[0]

        # delete patient profile first
        cur.execute(
            "DELETE FROM patient_profiles WHERE id = %s",
            (id,)
        )

        # then delete user
        cur.execute(
            "DELETE FROM users WHERE id = %s",
            (user_id,)
        )

        conn.commit()
        return jsonify({"success": True})

    except Exception as e:
        conn.rollback()
        print("DELETE PATIENT ERROR:", e)
        return jsonify({"error": "Database error"}), 500

    finally:
        cur.close()
        conn.close()

# =========================
# ATTENDANTS
# =========================
@admin_bp.route("/attendants", methods=["GET", "POST"])
def attendants():
    if "user_id" not in session or session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # =========================
    # CREATE ATTENDANT
    # =========================
    if request.method == "POST":
        d = request.json

        try:
            required_field(d.get("name"), "Full name")
            required_field(d.get("username"), "Username")
            required_field(d.get("password"), "Password")
            required_field(d.get("gender"), "Gender")
            required_field(d.get("dob"), "Date of birth")
            required_field(d.get("age"), "Age")
            required_field(d.get("code"), "Country code")
            required_field(d.get("contact"), "Contact number")
            required_field(d.get("address"), "Address")
            required_field(d.get("role"), "Role")
            required_field(d.get("shift"), "Shift")

            cur.execute(
                "SELECT 1 FROM users WHERE username = %s",
                (d["username"],)
            )
            if cur.fetchone():
                return jsonify({"error": "Username already exists"}), 409

            hashed = generate_password_hash(d["password"])

            # create user
            cur.execute("""
                INSERT INTO users (username, role, password_hash)
                VALUES (%s, 'attendant', %s)
                RETURNING id
            """, (d["username"].strip(), hashed))

            user_id = cur.fetchone()["id"]

            # create attendant profile
            cur.execute("""
    INSERT INTO attendants (
        user_id, full_name, gender, role, shift, phone_code, contact,
        date_of_birth, age, address, notes, email
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
""", (
    user_id,
    d["name"].strip(),
    d["gender"],
    d["role"],
    d["shift"],
    d["code"],
    d["contact"].strip(),
    d.get("dob"),          # optional
    int(d.get("age", 0)),  # default 0 if empty
    d.get("address"),      # optional
    d.get("notes"),
    d.get("email")         # optional
))


            conn.commit()
            return jsonify({"success": True})

        except ValueError as ve:
            conn.rollback()
            return jsonify({"error": str(ve)}), 400

        except Exception as e:
            conn.rollback()
            print("ATTENDANT SAVE ERROR:", e)
            return jsonify({"error": "Database error"}), 500

        finally:
            cur.close()
            conn.close()

    # =========================
    # LIST ATTENDANTS
    # =========================
    cur.execute("""
    SELECT
  id,
  full_name AS name,
  role,
  shift,
  phone_code || ' ' || contact AS contact,
  'Active' AS status
FROM attendants

    ORDER BY id DESC
""")


    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(rows)

@admin_bp.route("/attendants/<int:id>", methods=["DELETE"])
def delete_attendant(id):
    if "user_id" not in session or session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            "SELECT user_id FROM attendants WHERE id = %s",
            (id,)
        )
        row = cur.fetchone()

        if not row:
            return jsonify({"error": "Attendant not found"}), 404

        user_id = row[0]

        cur.execute("DELETE FROM attendants WHERE id = %s", (id,))
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))

        conn.commit()
        return jsonify({"success": True})

    except Exception as e:
        conn.rollback()
        print("DELETE ATTENDANT ERROR:", e)
        return jsonify({"error": "Database error"}), 500

    finally:
        cur.close()
        conn.close()


# =========================
# DOCTORS
# =========================
@admin_bp.route("/doctors", methods=["GET", "POST"])
def doctors():
    if "user_id" not in session or session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # =========================
    # CREATE DOCTOR
    # =========================
    if request.method == "POST":
        d = request.json

        try:
            required_field(d.get("name"), "Full name")
            required_field(d.get("username"), "Username")
            required_field(d.get("password"), "Password")
            required_field(d.get("gender"), "Gender")
            required_field(d.get("specialization"), "Specialization")
            required_field(d.get("department"), "Department")
            required_field(d.get("status"), "Status")
            required_field(d.get("code"), "Country code")
            required_field(d.get("phone"), "Contact number")

            # username unique
            cur.execute("SELECT 1 FROM users WHERE username=%s", (d["username"],))
            if cur.fetchone():
                return jsonify({"error": "Username already exists"}), 409

            hashed = generate_password_hash(d["password"])

            # create user
            cur.execute("""
                INSERT INTO users (username, role, password_hash)
                VALUES (%s,'doctor',%s)
                RETURNING id
            """, (d["username"].strip(), hashed))

            user_id = cur.fetchone()["id"]

            # create doctor profile
            cur.execute("""
                INSERT INTO doctors (
                    user_id, full_name, email, gender,
                    date_of_birth, age,
                    phone_code, contact, address,
                    specialization, department,
                    experience_years, status
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                user_id,
                d["name"].strip(),
                d.get("email"),
                d["gender"],
                d.get("dob"),
                int(d.get("age", 0)),
                d["code"],
                d["phone"],
                d.get("address"),
                d["specialization"],
                d["department"],
                int(d.get("experience", 0)),
                d["status"]
            ))

            conn.commit()
            return jsonify({"success": True})

        except ValueError as ve:
            conn.rollback()
            return jsonify({"error": str(ve)}), 400

        except Exception as e:
            conn.rollback()
            print("DOCTOR SAVE ERROR:", e)
            return jsonify({"error": "Database error"}), 500

        finally:
            cur.close()
            conn.close()

    # =========================
    # LIST DOCTORS
    # =========================
    cur.execute("""
    SELECT
        id,
        full_name AS name,
        specialization,
        department,
        experience_years,
        status
    FROM doctors
    ORDER BY id DESC
""")

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(rows)

@admin_bp.route("/doctors/<int:id>", methods=["DELETE"])
def delete_doctor(id):
    if "user_id" not in session or session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("SELECT user_id FROM doctors WHERE id=%s", (id,))
        row = cur.fetchone()

        if not row:
            return jsonify({"error": "Doctor not found"}), 404

        user_id = row[0]

        cur.execute("DELETE FROM doctors WHERE id=%s", (id,))
        cur.execute("DELETE FROM users WHERE id=%s", (user_id,))

        conn.commit()
        return jsonify({"success": True})

    except Exception as e:
        conn.rollback()
        print("DELETE DOCTOR ERROR:", e)
        return jsonify({"error": "Database error"}), 500

    finally:
        cur.close()
        conn.close()

# =========================
# APPOINTMENTS
# =========================
@admin_bp.route("/appointments", methods=["GET", "POST"])
def appointments():
    if "user_id" not in session or session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # =========================
    # CREATE / UPDATE
    # =========================
    if request.method == "POST":
        d = request.json

        try:
            # required
            if not d.get("patient"):
                raise ValueError("Patient required")
            if not d.get("doctor"):
                raise ValueError("Doctor required")
            if not d.get("date"):
                raise ValueError("Date required")
            if not d.get("time"):
                raise ValueError("Time required")

            # =========================
            # UPDATE
            # =========================
            if d.get("id"):
                cur.execute("""
                    UPDATE appointments
                    SET patient_id=%s,
                        doctor_id=%s,
                        appointment_date=%s,
                        appointment_time=%s,
                        status=%s,
                        reason=%s
                    WHERE id=%s
                """, (
                    d["patient"],
                    d["doctor"],
                    d["date"],
                    d["time"],
                    d.get("status", "Pending"),
                    d.get("reason"),
                    d["id"]
                ))

            # =========================
            # CREATE
            # =========================
            else:
                cur.execute("""
                    INSERT INTO appointments
                    (patient_id, doctor_id, appointment_date,
                     appointment_time, status, reason)
                    VALUES (%s,%s,%s,%s,%s,%s)
                """, (
                    d["patient"],
                    d["doctor"],
                    d["date"],
                    d["time"],
                    d.get("status", "Pending"),
                    d.get("reason")
                ))

            conn.commit()
            return jsonify({"success": True})

        except ValueError as ve:
            conn.rollback()
            return jsonify({"error": str(ve)}), 400

        except Exception as e:
            conn.rollback()
            print("APPOINTMENT SAVE ERROR:", e)
            return jsonify({"error": "Database error"}), 500

        finally:
            cur.close()
            conn.close()

    # =========================
    # LIST
    # =========================
    cur.execute("""
        SELECT
    a.id,
    a.patient_id,
    a.doctor_id,
    TO_CHAR(a.appointment_date, 'YYYY-MM-DD') AS date,
    TO_CHAR(a.appointment_time, 'HH24:MI') AS time,
    p.full_name AS patient,
    d.full_name AS doctor,
    a.reason,
    a.status
FROM appointments a
JOIN patient_profiles p ON a.patient_id = p.id
JOIN doctors d ON a.doctor_id = d.id
ORDER BY a.appointment_date DESC, a.appointment_time DESC
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return jsonify(rows if rows else [])
@admin_bp.route("/appointments/<int:id>", methods=["DELETE"])
def delete_appointment(id):
    if "user_id" not in session or session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("DELETE FROM appointments WHERE id=%s", (id,))
        conn.commit()
        return jsonify({"success": True})

    except Exception as e:
        conn.rollback()
        print("DELETE APPOINTMENT ERROR:", e)
        return jsonify({"error": "Database error"}), 500

    finally:
        cur.close()
        conn.close()



@admin_bp.route("/settings", methods=["POST"])
def settings():
    if not admin_required():
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db_connection()
    cur = conn.cursor()

    d = request.json
    user_id = session.get("user_id")

    try:
        email = d.get("email")
        password = d.get("password")

        # =========================
        # CHECK USERNAME EXISTS
        # =========================
        if email:
            cur.execute("""
                SELECT id FROM users
                WHERE username = %s AND id != %s
            """, (email, user_id))

            if cur.fetchone():
                return jsonify({"error": "Username already exists"}), 409

            # update username
            cur.execute("""
                UPDATE users
                SET username = %s
                WHERE id = %s
            """, (email, user_id))

        # =========================
        # UPDATE PASSWORD
        # =========================
        if password:
            hashed = generate_password_hash(password)
            cur.execute("""
                UPDATE users
                SET password_hash = %s
                WHERE id = %s
            """, (hashed, user_id))

        conn.commit()
        return jsonify({"success": True})

    except Exception as e:
        conn.rollback()
        print("SETTINGS ERROR:", e)
        return jsonify({"error": "Database error"}), 500

    finally:
        cur.close()
        conn.close()
