from flask import Blueprint, request, jsonify, session
import pandas as pd
import json
import bcrypt
from datetime import datetime
from config.database_config import get_db_connection
from models.sepsis_predictor import SepsisPredictor
from utils.helpers import Helpers
from chatbot.nlp_processor import NLPProcessor
from chatbot.response_generator import ResponseGenerator


nlp_processor = NLPProcessor()
response_generator = ResponseGenerator()


patient_bp = Blueprint("patient", __name__)

sepsis_predictor = SepsisPredictor()
helpers = Helpers()

nlp_loaded = False
model_loaded = False
feature_names = []



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
            INSERT INTO prediction_history (user_id, prediction, probability, risk_level)
            VALUES (%s, %s, %s, %s)
        """, (session["user_id"], result["prediction"], float(result["probability"]), result["risk_level"]))
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
    
@patient_bp.route("/patient/history")
def patient_history():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT prediction, probability, risk_level, timestamp
        FROM prediction_history 
        WHERE user_id=%s ORDER BY timestamp DESC
    """, (session["user_id"],))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    data = [
        {"prediction": r[0], "probability": r[1], "risk_level": r[2], "timestamp": r[3].isoformat()}
        for r in rows
    ]
    return jsonify(data)

@patient_bp.route("/patient/report")
def patient_report():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*), AVG(probability), MAX(risk_level)
        FROM prediction_history WHERE user_id=%s
    """, (session["user_id"],))
    total, avg, high = cur.fetchone()
    cur.close()
    conn.close()

    return jsonify({
        "total_predictions": total,
        "average_probability": round((avg or 0) * 100, 2),
        "highest_risk": high or "None"
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

@patient_bp.route("/patient/assistant", methods=["POST"])
def assistant():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    global nlp_loaded
    if not nlp_loaded:
        try:
            nlp_processor.load_intent_classifier()  # Load trained chatbot model
            nlp_loaded = True
        except Exception as e:
            return jsonify({"error": f"NLP model load failed: {e}"}), 500

    text = request.json.get("message", "")
    if not text:
        return jsonify({"error": "Empty message"}), 400

    try:
        intent = nlp_processor.extract_intent(text)
        response = response_generator.generate_response(
            intent=intent,
            entities={},
            context={},
            original_message=text
        )
        return jsonify({"reply": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@patient_bp.route("/patient/settings", methods=["POST"])
def settings_update():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    data = request.json
    return jsonify({"saved": True, "preferences": data})

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

    # 1. Get old hash
    cur.execute("SELECT password_hash FROM users WHERE id=%s", (session["user_id"],))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        return jsonify({"error": "User not found"}), 404

    stored_hash = row[0].encode("utf-8")

    # 2. Verify old password
    if not bcrypt.checkpw(current_password.encode("utf-8"), stored_hash):
        cur.close()
        conn.close()
        return jsonify({"error": "Current password is incorrect"}), 403

    # 3. Hash new password
    new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt())

    # 4. Update DB
    cur.execute(
        "UPDATE users SET password_hash=%s WHERE id=%s",
        (new_hash.decode("utf-8"), session["user_id"])
    )
    conn.commit()

    cur.close()
    conn.close()

    return jsonify({"success": True, "message": "Password updated successfully"})
