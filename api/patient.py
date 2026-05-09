from flask import Blueprint, request, jsonify, session, Response
import pandas as pd
import json
import os
import re
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
import bcrypt
from datetime import datetime
from config.database_config import get_db_connection
from models.sepsis_predictor import SepsisPredictor
from utils.helpers import Helpers
from chatbot.nlp_processor import NLPProcessor
from chatbot.response_generator import ResponseGenerator
from chatbot.sepsis_doc_agent import SepsisDocAgent
from utils.constants import CHATBOT_INTENTS
from flask import request
import hashlib
import base64
response_gen = ResponseGenerator()
doc_agent = SepsisDocAgent()

nlp_processor = NLPProcessor()
from werkzeug.security import check_password_hash, generate_password_hash


patient_bp = Blueprint("patient", __name__)

sepsis_predictor = SepsisPredictor()
helpers = Helpers()

nlp_loaded = False
model_loaded = False
feature_names = []



UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

@patient_bp.route("/patient/upload-lab", methods=["POST"])
def upload_lab():

    global model_loaded
    global feature_names

    # =========================
    # AUTH
    # =========================
    if "user_id" not in session:
        return jsonify({
            "error": "Not authenticated"
        }), 401

    # =========================
    # FILE CHECK
    # =========================
    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "Empty filename"
        }), 400

    # =========================
    # LOAD MODEL ONCE
    # =========================
    if not model_loaded:

        sepsis_predictor.load_model(
            "models/saved_models/sepsis_model.pkl"
        )

        with open(
            "models/saved_models/feature_names.json",
            "r"
        ) as f:

            feature_names = json.load(f)

        model_loaded = True

    # =========================
    # SAVE FILE
    # =========================
    filename = file.filename

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(filepath)

    # =========================
    # SAVE TO DATABASE
    # =========================
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO lab_reports
        (user_id, file_name, file_path)
        VALUES (%s, %s, %s)
    """, (
        session["user_id"],
        filename,
        filepath
    ))

    conn.commit()

    cur.close()
    conn.close()

    # =========================
    # EXTRACT TEXT
    # =========================
    extracted_text = ""

    try:

        # PDF
        if filename.lower().endswith(".pdf"):

            reader = PdfReader(filepath)

            for page in reader.pages:

                txt = page.extract_text()

                if txt:
                    extracted_text += txt + "\n"

        # IMAGE
        elif filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):

            image = Image.open(filepath)

            extracted_text = pytesseract.image_to_string(image)

        else:
            return jsonify({
                "error": "Unsupported file type"
            }), 400

    except Exception as e:

        return jsonify({
            "error": f"Text extraction failed: {str(e)}"
        }), 500

    # =========================
    # CLEAN TEXT
    # =========================
    extracted_text = extracted_text.replace("\n", " ")

    extracted_text = re.sub(
        r"\s+",
        " ",
        extracted_text
    )

    print("\n===== EXTRACTED TEXT =====")
    print(extracted_text)

    # =========================
    # SAFE EXTRACT FUNCTION
    # =========================
    def extract_number(patterns, text):

        if isinstance(patterns, str):
            patterns = [patterns]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:
                try:
                    return float(match.group(1))
                except:
                    pass

        return None

    # =========================
    # BLOOD PRESSURE
    # =========================
    bp_match = re.search(
        r"(?:bp[:\s]*)?([0-9]{2,3})\s*/\s*([0-9]{2,3})",
        extracted_text,
        re.IGNORECASE
    )

    systolic_bp = None
    diastolic_bp = None

    if bp_match:

        systolic_bp = float(bp_match.group(1))
        diastolic_bp = float(bp_match.group(2))

    # =========================
    # EXTRACT FEATURES
    # =========================
    patient_data = {

        "temperature": extract_number([
            r"temperature[:\s]*([0-9.]+)",
            r"temp[:\s]*([0-9.]+)"
        ], extracted_text),

        "heart_rate": extract_number([
            r"hr[:\s]*([0-9]+)",
            r"heart rate[:\s]*([0-9]+)"
        ], extracted_text),

        "respiratory_rate": extract_number([
            r"respiratory rate[:\s]*([0-9]+)",
            r"rr[:\s]*([0-9]+)"
        ], extracted_text),

        "wbc": extract_number([
            r"wbc[:\s]*([0-9.]+)",
            r"white blood cells?[:\s]*([0-9.]+)"
        ], extracted_text),

        "platelets": extract_number([
            r"platelets?[:\s]*([0-9.]+)"
        ], extracted_text),

        "lactate": extract_number([
            r"lactate[:\s]*([0-9.]+)"
        ], extracted_text),

        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp
    }

    # =========================
    # DEBUG
    # =========================
    print("\n===== EXTRACTED FEATURES =====")
    print(patient_data)

    # =========================
    # CHECK MISSING VALUES
    # =========================
    missing = []

    for k, v in patient_data.items():

        if v is None:
            missing.append(k)

    print("\n===== MISSING FEATURES =====")
    print(missing)

    # Too many missing values
    if len(missing) > 4:

        return jsonify({
            "error": (
                "Could not extract enough clinical data "
                f"from report. Missing: {missing}"
            )
        }), 400

    # =========================
    # FILL REMAINING NULLS
    # =========================
    for k in patient_data:

        if patient_data[k] is None:
            patient_data[k] = 0

    # =========================
    # ALIGN FEATURES
    # =========================
    fixed = {}

    for f in feature_names:
        fixed[f] = patient_data.get(f, 0)

    X = pd.DataFrame([fixed])

    print("\n===== MODEL INPUT =====")
    print(X)

    # =========================
    # PREDICTION
    # =========================
    try:

        result = sepsis_predictor.predict_single(
            X.iloc[0].to_dict(),
            feature_names,
            threshold=0.5
        )

    except Exception as e:

        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

    # =========================
    # RESPONSE
    # =========================
    return jsonify({

        "success": True,

        "prediction": int(
            result.get("prediction", 0)
        ),

        "risk_level": result.get(
            "risk_level",
            "LOW"
        ),

        "probability": float(
            result.get("probability", 0)
        ),

        "explanation":
            result.get(
                "explanation",
                "Prediction generated successfully."
            ),

        # OPTIONAL DEBUG INFO
        "extracted_text": extracted_text,

        "extracted_features": patient_data
    })


@patient_bp.route("/patient/lab-reports")
def get_lab_reports():

    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, file_name, uploaded_at
        FROM lab_reports
        WHERE user_id=%s
        ORDER BY uploaded_at DESC
    """, (session["user_id"],))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    reports = []

    for r in rows:
        reports.append({
            "id": r[0],
            "file_name": r[1],
            "uploaded_at": r[2].strftime("%Y-%m-%d %H:%M")
        })

    return jsonify(reports)

@patient_bp.route("/patient/submit_vitals", methods=["POST"])
def submit_vitals():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session["user_id"]
    data = request.json

    # 🔹 Extract vitals from request
    temperature = data.get("temperature")
    heart_rate = data.get("heart_rate")
    respiratory_rate = data.get("respiratory_rate")
    o2_saturation = data.get("o2_saturation")

    if not all([temperature, heart_rate, respiratory_rate, o2_saturation]):
        return jsonify({"error": "Please provide all vitals"}), 400

    # 🔹 Save vitals in DB
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO vitals (user_id, temperature, heart_rate, respiratory_rate, o2_saturation, source, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            temperature,
            heart_rate,
            respiratory_rate,
            o2_saturation,
            "manual",
            datetime.now()
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("❌ Error saving vitals:", str(e))
        return jsonify({"error": "Could not save vitals"}), 500

    return jsonify({"message": "Vitals submitted successfully", "vitals": data})

@patient_bp.route("/patient/predict", methods=["POST"])
def patient_predict():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    if session.get("role") != "patient":
        return jsonify({"error": "Unauthorized"}), 403

    try:
        global model_loaded, feature_names
        if not model_loaded:
            sepsis_predictor.load_model("models/saved_models/sepsis_model.pkl")
            with open("models/saved_models/feature_names.json") as f:
                feature_names = json.load(f)
            model_loaded = True

        data = request.json
        patient_features = helpers.prepare_patient_features(data)
        fixed = {f: patient_features.get(f, 0) for f in feature_names}
        X = pd.DataFrame([fixed])

        result = sepsis_predictor.predict_single(X.iloc[0].to_dict(), feature_names, threshold=0.5)

        # ⭐ SAVE TO DATABASE
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
    INSERT INTO prediction_history (
        user_id,
        prediction,
        probability,
        risk_level,
        heart_rate,
        respiratory_rate,
        blood_pressure,
        temperature,
        notes,
        timestamp
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
""", (
    session["user_id"],
    result["prediction"],
    float(result["probability"]),
    result["risk_level"],
    data.get("heart_rate"),
    data.get("respiratory_rate"),
    f"{data.get('systolic_bp')}/{data.get('diastolic_bp')}",
    data.get("temperature"),
    "AI Sepsis Risk Assessment",
    datetime.now()
))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "probability": float(result["probability"]),
            "risk_level": result["risk_level"],
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@patient_bp.route("/patient/profile", methods=["POST"])
def create_or_update_patient_profile():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    if session.get("role") != "patient":
        return jsonify({"error": "Unauthorized"}), 403

    user_id = session["user_id"]
    data = request.json

    conn = get_db_connection()
    cur = conn.cursor()

    # 🔎 Check existing profile
    cur.execute(
        "SELECT id FROM patient_profiles WHERE user_id=%s",
        (user_id,)
    )
    existing = cur.fetchone()

    if existing:
        # 🔄 UPDATE
        cur.execute("""
            UPDATE patient_profiles SET
                full_name=%s,
                gender=%s,
                date_of_birth=%s,
                age=%s,
                blood_group=%s,
                contact_country_code=%s,
                contact_number=%s,
                admission_type=%s,
                address=%s,
                updated_at=%s
            WHERE user_id=%s
        """, (
            data.get("full_name"),
            data.get("gender"),
            data.get("dob"),
            data.get("age"),
            data.get("blood_group"),
            data.get("contact_country_code"),
            data.get("contact_number"),
            data.get("admission_type"),
            data.get("address"),
            datetime.now(),
            user_id
        ))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"success": True, "message": "Profile updated"})

    else:
        # ➕ CREATE
        cur.execute("""
            INSERT INTO patient_profiles
            (user_id, full_name, gender, date_of_birth, age, blood_group,
             contact_country_code, contact_number, admission_type, address,
             created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            user_id,
            data.get("full_name"),
            data.get("gender"),
            data.get("dob"),
            data.get("age"),
            data.get("blood_group"),
            data.get("contact_country_code"),
            data.get("contact_number"),
            data.get("admission_type"),
            data.get("address"),
            datetime.now(),
            datetime.now()
        ))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"success": True, "message": "Profile created"})
    
@patient_bp.route("/patient/profile", methods=["GET"])
def get_patient_profile():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    if session.get("role") != "patient":
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT full_name, gender, date_of_birth, age, blood_group,
               contact_country_code, contact_number, admission_type, address
        FROM patient_profiles
        WHERE user_id=%s
    """, (session["user_id"],))

    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return jsonify({"exists": False})

    return jsonify({
        "exists": True,
        "profile": {
            "full_name": row[0],
            "gender": row[1],
            "dob": row[2],
            "age": row[3],
            "blood_group": row[4],
            "contact_country_code": row[5],
            "contact_number": row[6],
            "admission_type": row[7],
            "address": row[8],
        }
    })


@patient_bp.route("/patient/history")
def patient_history():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            heart_rate,
            respiratory_rate,
            blood_pressure,
            temperature,
            risk_level,
            notes,
            timestamp
        FROM prediction_history 
        WHERE user_id=%s 
        ORDER BY timestamp DESC
    """, (session["user_id"],))
    
    rows = cur.fetchall()
    cur.close()
    conn.close()

    data = [
        {
            "heart_rate": r[0],
            "respiratory_rate": r[1],
            "blood_pressure": r[2],
            "temperature": r[3],
            "risk_score": r[4],
            "notes": r[5],
            "timestamp": r[6].strftime("%Y-%m-%d %H:%M")
        }
        for r in rows
    ]

    return jsonify(data)
@patient_bp.route("/patient/report")
def patient_report():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session["user_id"]
    conn = get_db_connection()
    cur = conn.cursor()

    # ✅ 1. GET LATEST PREDICTION
    cur.execute("""
        SELECT prediction, probability, risk_level,
               heart_rate, respiratory_rate, blood_pressure, temperature,
               timestamp
        FROM prediction_history
        WHERE user_id=%s
        ORDER BY timestamp DESC LIMIT 1
    """, (user_id,))
    
    row = cur.fetchone()   # 🔥 FETCH HERE

    # ✅ 2. GET PROFILE
    cur.execute("""
        SELECT full_name, age
        FROM patient_profiles
        WHERE user_id=%s
    """, (user_id,))
    
    profile_row = cur.fetchone()   # 🔥 FETCH HERE

    # ❌ NO DATA CASE
    if not row:
        return jsonify({"error": "No data found"})

    prediction, prob, risk, hr, rr, bp, temp, ts = row
    prob = float(prob) if prob else 0

    # =========================
    # EXPLANATION ENGINE
    # =========================
    reasons = []

    if temp and temp > 38:
        reasons.append(f"High temperature ({temp}°C)")

    if rr and rr > 20:
        reasons.append(f"Elevated respiratory rate ({rr})")

    if hr and hr > 100:
        reasons.append(f"High heart rate ({hr})")

    if bp and isinstance(bp, str) and "/" in bp:
     try:
        parts = bp.split("/")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            sys = int(parts[0])
            dia = int(parts[1])

            if sys > 140 or dia > 90:
                reasons.append(f"High blood pressure ({bp})")
     except Exception as e:
        print("BP parsing error:", e)

    if not reasons:
        reasons.append("Vitals are within acceptable range")

    risk_clean = risk.upper().strip()


    if "HIGH" in risk_clean:
     explanation = "Critical condition: " + " & ".join(reasons)
    elif "MODERATE" in risk_clean:
     explanation = "Moderate concern: " + " & ".join(reasons)
    else:
     explanation = "Stable condition: " + " & ".join(reasons)

    clinical_text = f"""
Respiratory: {'High' if rr and rr > 20 else 'Normal'}  
Heart Rate: {'High' if hr and hr > 100 else 'Normal'}  
Temperature: {'Fever detected' if temp and temp > 38 else 'Normal'}  
Blood Pressure: {bp if bp else 'Normal'}  

Overall: {explanation}
"""

    cur.close()
    conn.close()

    return jsonify({
    "total_predictions": 1,
    "latest": {
        "prediction": prediction,
        "probability": prob,
        "risk_level": risk,
        "heart_rate": hr,
        "respiratory_rate": rr,
        "blood_pressure": bp,
        "temperature": temp,
        "timestamp": ts.isoformat() if ts else None,
        "explanation": explanation,
        "clinical_observations": clinical_text 
    },
        "profile": {
            "full_name": profile_row[0] if profile_row else None,
            "age": profile_row[1] if profile_row else None
        }
    })
@patient_bp.route("/patient/simulate")
def simulate():
    import random
    return jsonify({
        "temperature": round(random.uniform(36, 40), 1),
        "heart_rate": random.randint(60, 140),
        "respiratory_rate": random.randint(10, 32),
        "o2_saturation": random.randint(80, 100)
    })

def detect_intent(message: str):
    msg = message.lower().strip()

    # Greeting
    if any(w in msg for w in ["hi", "hello", "hey"]):
        return CHATBOT_INTENTS['GREETING']

    # Direct definition question
    if "what is sepsis" in msg or "define sepsis" in msg:
        return "DOC_QUERY"

    # Symptoms
    if any(w in msg for w in ["symptom", "sign", "feel", "fever", "pain"]):
        return CHATBOT_INTENTS['SYMPTOMS']

    # Prevention
    if any(w in msg for w in ["prevent", "prevention", "avoid", "protection", "stop sepsis"]):
        return CHATBOT_INTENTS['PREVENTION']

    # Causes  (keep BEFORE generic why/reason logic)
    if any(w in msg for w in ["cause of sepsis", "causes of sepsis", "why does sepsis happen", "how sepsis happens"]):
        return "CAUSES"

    # Treatment
    if any(w in msg for w in ["treat", "treatment", "cure", "medicine", "antibiotic"]):
        return CHATBOT_INTENTS['TREATMENT']

    # Risk prediction
    if any(w in msg for w in ["risk", "risks", "chance", "probability"]):
        return CHATBOT_INTENTS['SEPSIS_RISK']

    # If message is long → send to document QA
    if len(msg.split()) > 4:
        return "DOC_QUERY"

    # Otherwise fallback
    return CHATBOT_INTENTS['HELP']



@patient_bp.route("/patient/assistant", methods=["POST"])
def assistant():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    text = request.json.get("message", "").strip()
    if not text:
        return jsonify({"reply": "Please enter a question."})

    # Detect user intent
    intent = detect_intent(text)

    # 🟢 Direct response intents (no document search)
    if intent in [
        CHATBOT_INTENTS['GREETING'],
        CHATBOT_INTENTS['SYMPTOMS'],
        CHATBOT_INTENTS['PREVENTION'],
        CHATBOT_INTENTS['TREATMENT'],
        CHATBOT_INTENTS['HELP'],
        CHATBOT_INTENTS['GOODBYE'],
        "CAUSES"
    ]:
        response = response_gen.generate_response(
            intent=intent,
            original_message=text
        )
        return jsonify({
            "reply": response["response"],
            "intent": intent,
            "suggestions": response["suggestions"]
        })

    # 🔵 Document-based answers (only DOC_QUERY)
    if intent == "DOC_QUERY":
        try:
            doc_answer = doc_agent.answer(text)
        except Exception as e:
            print("❌ Document agent error:", str(e))
            doc_answer = None

        # Fallback if document agent fails
        if not doc_answer or "could not find information" in doc_answer.lower():
            fallback = response_gen.generate_response(
                intent=CHATBOT_INTENTS['HELP'],
                original_message=text
            )
            return jsonify({
                "reply": fallback["response"],
                "intent": "fallback",
                "suggestions": fallback["suggestions"]
            })

        return jsonify({
            "reply": doc_answer,
            "intent": "DOC_QUERY",
            "suggestions": response_gen._generate_suggestions(
                CHATBOT_INTENTS['SYMPTOMS']
            )
        })

    # 🔁 Catch-all fallback for any other cases
    fallback = response_gen.generate_fallback_response(text)
    return jsonify({
        "reply": fallback["response"],
        "intent": fallback["data"]["intent"],
        "suggestions": fallback["suggestions"]
    })


@patient_bp.route("/patient/settings", methods=["POST"])
def settings_update():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.json

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE users
        SET email_alerts=%s,
            sms_alerts=%s,
            weekly_report=%s
        WHERE id=%s
    """, (
        data.get("email_alerts"),
        data.get("sms_alerts"),
        data.get("weekly_report"),
        session["user_id"]
    ))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"saved": True})


@patient_bp.route("/patient/update-password", methods=["POST"])
def update_password():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.json
    current_password = data.get("current_password")
    new_password = data.get("new_password")

    if not current_password or not new_password:
        return jsonify({"error": "Missing fields"}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT password_hash FROM users WHERE id=%s", (session["user_id"],))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        return jsonify({"error": "User not found"}), 404

    stored_hash = row[0]

    # ✅ CORRECT CHECK
    if not check_password_hash(stored_hash, current_password):
        cur.close()
        conn.close()
        return jsonify({"error": "Current password is incorrect"}), 403

    # ✅ CORRECT HASHING
    new_hash = generate_password_hash(new_password)

    cur.execute(
        "UPDATE users SET password_hash=%s WHERE id=%s",
        (new_hash, session["user_id"])
    )
    conn.commit()

    cur.close()
    conn.close()

    return jsonify({"success": True, "message": "Password updated successfully"})

@patient_bp.route("/patient/update-email", methods=["POST"])
def update_email():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.json
    current_email = data.get("current_email", "").strip()
    new_email = data.get("new_email", "").strip()
    confirm_new_email = data.get("confirm_new_email", "").strip()

    if not current_email or not new_email or not confirm_new_email:
        return jsonify({"error": "Current email, new email, and confirmation are required"}), 400

    if new_email != confirm_new_email:
        return jsonify({"error": "New email and confirmation do not match"}), 400

    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 1️⃣ Get email from DB
            cur.execute("SELECT email FROM users WHERE id=%s", (session["user_id"],))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "User not found"}), 404

            stored_email = row[0]

            # 2️⃣ Verify current email
            if current_email != stored_email:
                return jsonify({"error": "Current email does not match"}), 403

            # 3️⃣ Check if new email is already used by someone else
            cur.execute("SELECT id FROM users WHERE email=%s", (new_email,))
            if cur.fetchone():
                return jsonify({"error": "New email already in use"}), 409

            # 4️⃣ Update email
            cur.execute(
                "UPDATE users SET email=%s WHERE id=%s",
                (new_email, session["user_id"])
            )
        conn.commit()
    finally:
        conn.close()

    return jsonify({"success": True, "message": "Email updated successfully"})



@patient_bp.route("/patient/update-username", methods=["POST"])
def update_username():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.json
    current_username = data.get("current_username")
    new_username = data.get("new_username")

    if not current_username or not new_username:
        return jsonify({"error": "Missing fields"}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    # 🔎 Verify current username
    cur.execute(
        "SELECT username FROM users WHERE id=%s",
        (session["user_id"],)
    )
    stored_username = cur.fetchone()[0]

    if current_username != stored_username:
        cur.close()
        conn.close()
        return jsonify({"error": "Current username is incorrect"}), 403

    # 🔎 Check duplicate username
    cur.execute(
        "SELECT id FROM users WHERE username=%s",
        (new_username,)
    )
    if cur.fetchone():
        cur.close()
        conn.close()
        return jsonify({"error": "Username already taken"}), 409

    # 🔄 Update username
    cur.execute(
        "UPDATE users SET username=%s WHERE id=%s",
        (new_username, session["user_id"])
    )

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"success": True})

@patient_bp.route("/patient/export", methods=["GET"])
def export_data():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT id, heart_rate, bp, created_at
    FROM vitals
    WHERE user_id=%s
    """, (session["user_id"],))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id","heart_rate","bp","created_at"])
    writer.writerows(rows)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=data.csv"}
    )
