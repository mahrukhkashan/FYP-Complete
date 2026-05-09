#!/usr/bin/env python3
"""
==============================================================
FLASK APPLICATION  –  Explainable Multimodal AI-based
Sepsis Prediction and Assistance System
==============================================================

Start with:
    cd Fyp-part1-main-main
    python run.py
    # or directly:
    python -m flask --app api/app.py run --port 5000

Default admin login: admin / admin123
"""

import os
import sys
import json
import pickle
import logging
import traceback
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import pandas as pd
import joblib

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, g
)
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# ─── Blueprints ───────────────────────────────────────────────────────────────
from api.auth    import auth_bp
from api.routes  import api_bp
from api.patient import patient_bp
from api.doctor import doctor_bp
from api.admin import admin_bp
from api.attendant import attendant_bp



# ─── Internal modules ─────────────────────────────────────────────────────────
from config.config          import Config
from config.database_config import get_db_connection
from chatbot.nlp_processor  import NLPProcessor
from chatbot.response_generator import ResponseGenerator

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("sepsis_app")

# ─────────────────────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False
)
app.config.from_object(Config)
CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": [
        "http://127.0.0.1:5000",
        "http://localhost:5000"
    ]}}
)

app.register_blueprint(auth_bp,    url_prefix="/auth")
app.register_blueprint(api_bp)
app.register_blueprint(patient_bp)
app.register_blueprint(doctor_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(attendant_bp)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL ML STATE  (loaded once at startup, reused across requests)
# ─────────────────────────────────────────────────────────────────────────────
_state = {
    "model":          None,   # best sklearn/xgb model object
    "model_name":     None,   # e.g. "random_forest"
    "feature_names":  [],     # ordered list of feature names
    "imputer":        None,   # sklearn SimpleImputer
    "shap_explainer": None,   # shap.TreeExplainer / KernelExplainer
    "model_ready":    False,
    "model_metrics":  {},
}

MODEL_DIR = "models/saved_models"

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING HELPERS  (must mirror train_model.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_single_patient(raw: dict) -> dict:
    """
    Derive all engineered features from a raw dict of patient vitals/labs.
    Returns a flat dict of floats (same columns as X in training).
    Nulls are represented as np.nan.
    """
    def f(key, default=np.nan):
        v = raw.get(key)
        try:
            return float(v) if v not in (None, "", "null") else default
        except (ValueError, TypeError):
            return default

    age           = f("age")
    gender        = str(raw.get("gender", "")).upper()
    admission_type= str(raw.get("admission_type", "")).upper()
    heart_rate    = f("heart_rate")
    temperature   = f("temperature")
    resp_rate     = f("respiratory_rate")
    systolic_bp   = f("systolic_bp")
    diastolic_bp  = f("diastolic_bp")
    spo2          = f("spo2")
    wbc           = f("wbc")
    lactate       = f("lactate")
    creatinine    = f("creatinine")
    platelets     = f("platelets")
    bilirubin     = f("bilirubin")
    glucose       = f("glucose")
    sodium        = f("sodium")
    potassium     = f("potassium")
    bicarbonate   = f("bicarbonate")
    los_hours     = f("los_hours")

    # MAP
    _map = (diastolic_bp + (systolic_bp - diastolic_bp) / 3
            if not (np.isnan(systolic_bp) or np.isnan(diastolic_bp)) else np.nan)

    # Binary flags
    def safe_bool(condition):
        try:
            return float(bool(condition))
        except Exception:
            return np.nan

    hr_abn   = safe_bool(not np.isnan(heart_rate)     and (heart_rate < 60 or heart_rate > 100))
    temp_abn = safe_bool(not np.isnan(temperature)    and (temperature < 36.0 or temperature > 38.3))
    rr_abn   = safe_bool(not np.isnan(resp_rate)      and (resp_rate < 12 or resp_rate > 20))
    hypoxia  = safe_bool(not np.isnan(spo2)           and spo2 < 94)
    hypotens = safe_bool(not np.isnan(_map)           and _map < 65)
    wbc_abn  = safe_bool(not np.isnan(wbc)            and (wbc < 4 or wbc > 12))
    lac_high = safe_bool(not np.isnan(lactate)        and lactate > 2.0)
    lac_sev  = safe_bool(not np.isnan(lactate)        and lactate > 4.0)
    aki      = safe_bool(not np.isnan(creatinine)     and creatinine > 1.5)
    thrombo  = safe_bool(not np.isnan(platelets)      and platelets < 150)

    # SIRS
    sirs = 0.0
    if not np.isnan(temperature):
        sirs += float(temperature > 38.0 or temperature < 36.0)
    if not np.isnan(heart_rate):
        sirs += float(heart_rate > 90)
    if not np.isnan(resp_rate):
        sirs += float(resp_rate > 20)
    if not np.isnan(wbc):
        sirs += float(wbc > 12 or wbc < 4)
    meets_sirs = float(sirs >= 2)

    # qSOFA
    qsofa = 0.0
    if not np.isnan(resp_rate):
        qsofa += float(resp_rate >= 22)
    if not np.isnan(systolic_bp):
        qsofa += float(systolic_bp <= 100)
    high_qsofa = float(qsofa >= 2)

    # SOFA sub-scores
    def sofa_coag(p):
        if np.isnan(p): return np.nan
        if p < 20:   return 4.0
        if p < 50:   return 3.0
        if p < 100:  return 2.0
        if p < 150:  return 1.0
        return 0.0

    def sofa_renal(c):
        if np.isnan(c): return np.nan
        if c < 1.2:  return 0.0
        if c < 2.0:  return 1.0
        if c < 3.5:  return 2.0
        if c < 5.0:  return 3.0
        return 4.0

    def sofa_liver(b):
        if np.isnan(b): return np.nan
        if b < 1.2:  return 0.0
        if b < 2.0:  return 1.0
        if b < 6.0:  return 2.0
        if b < 12.0: return 3.0
        return 4.0

    def age_group(a):
        if np.isnan(a): return np.nan
        if a < 30: return 0.0
        if a < 50: return 1.0
        if a < 65: return 2.0
        if a < 80: return 3.0
        return 4.0

    organ_cols = [lac_sev, aki, thrombo, hypotens, hypoxia]
    organ_dysfunction = float(sum(v for v in organ_cols if not np.isnan(v)) >= 2)

    features = {
        # Demographics
        "age":             age,
        "gender_M":        float(gender == "M") if gender in ("M", "F") else np.nan,
        "is_emergency":    float(admission_type in ("EMERGENCY", "URGENT")),
        "age_group":       age_group(age),
        # Vitals
        "heart_rate":      heart_rate,
        "temperature":     temperature,
        "respiratory_rate": resp_rate,
        "systolic_bp":     systolic_bp,
        "diastolic_bp":    diastolic_bp,
        "spo2":            spo2,
        "map":             _map,
        # Labs
        "wbc":             wbc,
        "lactate":         lactate,
        "creatinine":      creatinine,
        "platelets":       platelets,
        "bilirubin":       bilirubin,
        "glucose":         glucose,
        "sodium":          sodium,
        "potassium":       potassium,
        "bicarbonate":     bicarbonate,
        "los_hours":       los_hours,
        # Binary flags
        "hr_abnormal":     hr_abn,
        "temp_abnormal":   temp_abn,
        "rr_abnormal":     rr_abn,
        "hypoxia":         hypoxia,
        "hypotension":     hypotens,
        "wbc_abnormal":    wbc_abn,
        "lactate_high":    lac_high,
        "lactate_severe":  lac_sev,
        "aki":             aki,
        "thrombocytopenia": thrombo,
        # Scores
        "sirs_score":      sirs,
        "meets_sirs":      meets_sirs,
        "qsofa_score":     qsofa,
        "high_qsofa":      high_qsofa,
        "sofa_coag":       sofa_coag(platelets),
        "sofa_renal":      sofa_renal(creatinine),
        "sofa_liver":      sofa_liver(bilirubin),
        "organ_dysfunction": organ_dysfunction,
    }
    return features


def build_feature_vector(raw: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame aligned to the training feature_names.
    Missing engineered features default to 0 (model imputer handles NaNs).
    """
    engineered = engineer_single_patient(raw)
    feature_names = _state["feature_names"]
    row = {f: engineered.get(f, 0.0) for f in feature_names}
    df  = pd.DataFrame([row], columns=feature_names)
    # Apply imputer if available
    if _state["imputer"] is not None:
        try:
            df = pd.DataFrame(
                _state["imputer"].transform(df),
                columns=feature_names
            )
        except Exception:
            df = df.fillna(0.0)
    return df


def risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def risk_color(level: str) -> str:
    return {"High Risk": "#e74c3c", "Medium Risk": "#e67e22", "Low Risk": "#27ae60"}.get(level, "#888")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model_artifacts(force: bool = False) -> bool:
    """Load saved model files into _state. Returns True on success."""
    if _state["model_ready"] and not force:
        return True

    try:
        model_path = os.path.join(MODEL_DIR, "sepsis_model.pkl")
        fn_path    = os.path.join(MODEL_DIR, "feature_names.json")
        fe_path    = os.path.join(MODEL_DIR, "feature_engineer.pkl")
        shap_path  = os.path.join(MODEL_DIR, "shap_explainer.pkl")

        if not os.path.exists(model_path):
            log.warning("No saved model found – run train_model.py first")
            return False

        artifact = joblib.load(model_path)
        _state["model"]        = artifact["model"]
        _state["model_name"]   = artifact.get("model_name", "unknown")
        _state["model_metrics"]= artifact.get("all_results", {})

        if os.path.exists(fn_path):
            with open(fn_path) as f:
                _state["feature_names"] = json.load(f)
        else:
            log.error("feature_names.json missing")
            return False

        if os.path.exists(fe_path):
            with open(fe_path, "rb") as f:
                bundle = pickle.load(f)
            _state["imputer"] = bundle.get("imputer")

        if os.path.exists(shap_path):
            with open(shap_path, "rb") as f:
                _state["shap_explainer"] = pickle.load(f)
            log.info("SHAP explainer loaded")

        _state["model_ready"] = True
        log.info(
            f"Model loaded: {_state['model_name']} | "
            f"{len(_state['feature_names'])} features"
        )
        return True

    except Exception as exc:
        log.error(f"load_model_artifacts failed: {exc}")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.is_json:
                return jsonify({"error": "Not authenticated"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if "user_id" not in session:
                return jsonify({"error": "Not authenticated"}), 401
            if session.get("role") not in roles:
                return jsonify({"error": "Forbidden"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# BASIC PAGE ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def signup_page():
    return render_template("index.html")


@app.route("/login")
def login_page():
    return render_template("login.html")


@app.route("/dashboard")

@app.route("/dashboard")
@login_required
def dashboard():
    role = session.get("role", "").lower()

    if role == "doctor":
        return render_template("doctorDashboard.html")
    elif role == "admin":
        return render_template("admin_dashboard.html")
    elif role == "attendant":
        return render_template("attendant_dashboard.html")
    elif role == "patient":
        return render_template("patientDashboard.html")
    else:
        session.clear()
        return redirect(url_for("login_page"))

@app.route("/forgot-password")
def forgot_password():
    return render_template("forgotpassword.html")


# ─────────────────────────────────────────────────────────────────────────────
# ══  PREDICTION  ══════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    """
    Accepts JSON body with raw clinical values.
    Returns prediction + probability + risk level.

    Example payload:
    {
        "age": 65, "gender": "M", "admission_type": "EMERGENCY",
        "heart_rate": 112, "temperature": 38.9, "respiratory_rate": 24,
        "systolic_bp": 88, "diastolic_bp": 55, "spo2": 91,
        "wbc": 16.5, "lactate": 3.2, "creatinine": 1.8,
        "platelets": 110, "bilirubin": 2.1
    }
    """
    if not load_model_artifacts():
        return jsonify({
            "error": "Model not ready. Please run train_model.py first."
        }), 503

    try:
        raw = request.get_json(force=True, silent=True) or {}
        if not raw:
            return jsonify({"error": "No JSON body received"}), 400

        X = build_feature_vector(raw)

        prob  = float(_state["model"].predict_proba(X)[0, 1])
        pred  = int(prob >= 0.5)
        level = risk_level(prob)

        # SHAP explanation (top-5 features)
        explanation = []
        if _state["shap_explainer"] is not None:
            try:
                sv = _state["shap_explainer"].shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1]   # class-1 SHAP values
                sv = np.array(sv).flatten()
                feat_names = _state["feature_names"]
                top_idx = np.argsort(np.abs(sv))[::-1][:5]
                explanation = [
                    {
                        "feature":      feat_names[i],
                        "value":        float(X.iloc[0, i]),
                        "shap_value":   float(sv[i]),
                        "contribution": "increases" if sv[i] > 0 else "decreases",
                    }
                    for i in top_idx
                ]
            except Exception as shap_err:
                log.warning(f"SHAP failed: {shap_err}")

        # Save prediction to DB
        _save_prediction_to_db(raw, prob, pred, level, explanation)

        return jsonify({
            "success":     True,
            "prediction":  pred,
            "probability": round(prob * 100, 2),   # percentage
            "risk_level":  level,
            "risk_color":  risk_color(level),
            "explanation": explanation,
            "timestamp":   datetime.now().isoformat(),
        })

    except Exception as exc:
        log.error(f"/predict error: {exc}")
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


def _save_prediction_to_db(raw: dict, prob: float, pred: int, level: str, explanation: list):
    """Persist prediction record. Non-fatal – errors are just logged."""
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO predictions
                (patient_id, clinician_id, risk_probability, risk_level,
                 prediction_result, confidence_score, model_version,
                 input_features, explanation)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            session.get("patient_db_id"),
            session.get("user_id"),
            round(prob, 4),
            level,
            bool(pred),
            round(prob, 4),
            _state["model_name"],
            json.dumps(raw),
            json.dumps(explanation),
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        log.warning(f"Could not save prediction to DB: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# ══  EXPLANATION  =============================================================
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/explain", methods=["POST"])
@login_required
def explain():
    """
    Return SHAP-based explanation for a given prediction.
    Accepts same JSON as /predict.
    """
    if not load_model_artifacts():
        return jsonify({"error": "Model not ready"}), 503

    if _state["shap_explainer"] is None:
        return jsonify({"error": "SHAP explainer not available"}), 503

    try:
        raw = request.get_json(force=True, silent=True) or {}
        X   = build_feature_vector(raw)

        sv = _state["shap_explainer"].shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.array(sv).flatten()

        feat_names = _state["feature_names"]
        all_effects = [
            {
                "feature":      feat_names[i],
                "value":        float(X.iloc[0, i]),
                "shap_value":   float(sv[i]),
                "contribution": "increases" if sv[i] > 0 else "decreases",
            }
            for i in range(len(feat_names))
        ]
        # Sort by absolute SHAP value
        all_effects.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # Base value
        try:
            base_value = float(_state["shap_explainer"].expected_value)
            if isinstance(_state["shap_explainer"].expected_value, (list, np.ndarray)):
                base_value = float(_state["shap_explainer"].expected_value[1])
        except Exception:
            base_value = 0.0

        return jsonify({
            "success":       True,
            "base_value":    round(base_value, 4),
            "feature_effects": all_effects[:15],   # top 15
        })

    except Exception as exc:
        log.error(f"/explain error: {exc}")
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ══  PATIENT DETAIL PREDICTION  ══════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/patient/<int:subject_id>/predict", methods=["GET", "POST"])
@login_required
def predict_patient(subject_id: int):
    """
    Fetch the latest vitals + labs for a MIMIC patient from PostgreSQL
    and run the model. Returns a JSON prediction.
    """
    if not load_model_artifacts():
        return jsonify({"error": "Model not ready"}), 503

    try:
        conn = get_db_connection()
        cur  = conn.cursor()

        # Latest vitals (chartevents)
        vitals_sql = """
        SELECT
            MAX(CASE WHEN di.label ILIKE '%%heart rate%%'               THEN ce.valuenum END) AS heart_rate,
            MAX(CASE WHEN di.label ILIKE '%%blood pressure%%systolic%%'  THEN ce.valuenum END) AS systolic_bp,
            MAX(CASE WHEN di.label ILIKE '%%blood pressure%%diastolic%%' THEN ce.valuenum END) AS diastolic_bp,
            MAX(CASE WHEN di.label ILIKE '%%temperature%%'               THEN ce.valuenum END) AS temperature_raw,
            MAX(CASE WHEN di.label ILIKE '%%respiratory rate%%'          THEN ce.valuenum END) AS respiratory_rate,
            MAX(CASE WHEN di.label ILIKE '%%oxygen saturation%%'         THEN ce.valuenum END) AS spo2
        FROM chartevents ce
        JOIN d_items di ON ce.itemid = di.itemid
        WHERE ce.subject_id = %s
          AND ce.valuenum IS NOT NULL AND ce.valuenum > 0
        """
        cur.execute(vitals_sql, (subject_id,))
        vitals_row = cur.fetchone()

        # Latest labs (labevents)
        labs_sql = """
        SELECT
            MAX(CASE WHEN dli.label ILIKE '%%white blood%%'  THEN le.valuenum END) AS wbc,
            MAX(CASE WHEN dli.label ILIKE '%%lactate%%'      THEN le.valuenum END) AS lactate,
            MAX(CASE WHEN dli.label ILIKE '%%creatinine%%'   THEN le.valuenum END) AS creatinine,
            MAX(CASE WHEN dli.label ILIKE '%%platelet%%'     THEN le.valuenum END) AS platelets,
            MAX(CASE WHEN dli.label ILIKE '%%bilirubin%%'    THEN le.valuenum END) AS bilirubin,
            MAX(CASE WHEN dli.label ILIKE '%%glucose%%'      THEN le.valuenum END) AS glucose
        FROM labevents le
        JOIN d_labitems dli ON le.itemid = dli.itemid
        WHERE le.subject_id = %s
          AND le.valuenum IS NOT NULL AND le.valuenum > 0
        """
        cur.execute(labs_sql, (subject_id,))
        labs_row = cur.fetchone()

        # Demographics
        demo_sql = """
        SELECT p.gender,
               EXTRACT(YEAR FROM AGE(a.admittime, p.dob))::int AS age,
               a.admission_type
        FROM patients p
        JOIN admissions a ON p.subject_id = a.subject_id
        WHERE p.subject_id = %s
        ORDER BY a.admittime DESC LIMIT 1
        """
        cur.execute(demo_sql, (subject_id,))
        demo_row = cur.fetchone()

        cur.close()
        conn.close()

        raw = {}
        if demo_row:
            raw.update({
                "gender":         demo_row[0] or "",
                "age":            demo_row[1],
                "admission_type": demo_row[2] or "",
            })
        if vitals_row:
            temp_raw = vitals_row[3]
            if temp_raw and temp_raw > 50:   # °F → °C
                temp_raw = (temp_raw - 32) * 5 / 9
            raw.update({
                "heart_rate":      vitals_row[0],
                "systolic_bp":     vitals_row[1],
                "diastolic_bp":    vitals_row[2],
                "temperature":     temp_raw,
                "respiratory_rate":vitals_row[4],
                "spo2":            vitals_row[5],
            })
        if labs_row:
            raw.update({
                "wbc":        labs_row[0],
                "lactate":    labs_row[1],
                "creatinine": labs_row[2],
                "platelets":  labs_row[3],
                "bilirubin":  labs_row[4],
                "glucose":    labs_row[5],
            })

        if not raw:
            return jsonify({"error": f"No data found for patient {subject_id}"}), 404

        X     = build_feature_vector(raw)
        prob  = float(_state["model"].predict_proba(X)[0, 1])
        pred  = int(prob >= 0.5)
        level = risk_level(prob)

        return jsonify({
            "success":        True,
            "subject_id":     subject_id,
            "prediction":     pred,
            "probability":    round(prob * 100, 2),
            "risk_level":     level,
            "risk_color":     risk_color(level),
            "input_features": {k: v for k, v in raw.items() if v is not None},
            "timestamp":      datetime.now().isoformat(),
        })

    except Exception as exc:
        log.error(f"/patient/{subject_id}/predict error: {exc}")
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ══  CHATBOT  ═════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

nlp_processor      = NLPProcessor()
response_generator = ResponseGenerator()


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data    = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    try:
        intent   = nlp_processor.extract_intent(message)
        entities = nlp_processor.extract_entities(message)
        response = response_generator.generate_response(
            intent=intent,
            entities=entities,
            context={"role": session.get("role", "patient")},
            original_message=message,
        )
        return jsonify(response)
    except Exception as exc:
        log.error(f"/chat error: {exc}")
        return jsonify({"reply": "I'm sorry, I encountered an error. Please try again.", "intent": "error"}), 200


# ─────────────────────────────────────────────────────────────────────────────
# ══  ADMIN – TRAIN MODEL  ═════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/train_model", methods=["POST"])
@role_required("admin")
def train_model_endpoint():
    """
    Trigger model re-training from the admin dashboard.
    Runs synchronously (could be made async with Celery for production).
    """
    try:
        # Import here to avoid circular import issues
        import importlib
        import train_model as tm
        importlib.reload(tm)         # always use latest version
        tm.train_and_save_model()
        load_model_artifacts(force=True)
        return jsonify({"success": True, "message": "Model retrained and reloaded."})
    except Exception as exc:
        log.error(f"/train_model error: {exc}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ══  MODEL INFO  ══════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/model/info", methods=["GET"])
@login_required
def model_info():
    if not load_model_artifacts():
        return jsonify({"model_ready": False, "message": "Run train_model.py first"})

    # Feature importance
    fi_data = []
    model = _state["model"]
    feat  = _state["feature_names"]
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        fi_data = sorted(
            [{"feature": feat[i], "importance": round(float(importance[i]), 4)}
             for i in range(len(feat))],
            key=lambda x: x["importance"], reverse=True
        )[:15]

    return jsonify({
        "model_ready":       _state["model_ready"],
        "model_name":        _state["model_name"],
        "feature_count":     len(feat),
        "shap_available":    _state["shap_explainer"] is not None,
        "feature_importance":fi_data,
        "all_model_metrics": _state["model_metrics"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# ══  PREDICTION HISTORY  ══════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predictions/history", methods=["GET"])
@login_required
def prediction_history():
    try:
        conn = get_db_connection()
        cur  = conn.cursor()

        if session.get("role") == "admin":
            cur.execute("""
                SELECT p.id, p.patient_id, p.clinician_id,
                       p.prediction_timestamp, p.risk_probability,
                       p.risk_level, p.prediction_result, p.model_version
                FROM predictions p
                ORDER BY p.prediction_timestamp DESC
                LIMIT 50
            """)
        else:
            cur.execute("""
                SELECT id, patient_id, clinician_id,
                       prediction_timestamp, risk_probability,
                       risk_level, prediction_result, model_version
                FROM predictions
                WHERE clinician_id = %s
                ORDER BY prediction_timestamp DESC
                LIMIT 20
            """, (session["user_id"],))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        history = [
            {
                "id":          r[0],
                "patient_id":  r[1],
                "clinician_id":r[2],
                "timestamp":   r[3].isoformat() if r[3] else None,
                "probability": round(float(r[4]) * 100, 1) if r[4] else None,
                "risk_level":  r[5],
                "prediction":  r[6],
                "model":       r[7],
            }
            for r in rows
        ]
        return jsonify({"success": True, "history": history})

    except Exception as exc:
        log.error(f"/predictions/history error: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ══  SYSTEM STATUS  ═══════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/system/status", methods=["GET"])
@login_required
def system_status():
    db_ok = False
    try:
        conn = get_db_connection()
        conn.close()
        db_ok = True
    except Exception:
        pass

    return jsonify({
        "api_status":     "Online",
        "database_status":"Connected" if db_ok else "Disconnected",
        "model_status":   "Loaded" if _state["model_ready"] else "Not loaded",
        "model_name":     _state["model_name"] or "—",
        "shap_status":    "Available" if _state["shap_explainer"] else "Not available",
        "last_updated":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


# ─────────────────────────────────────────────────────────────────────────────
# ══  ADMIN: CREATE ADMIN USER  ════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def ensure_admin_user():
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = %s", ("admin",))
        if not cur.fetchone():
            cur.execute("""
                INSERT INTO users (username, email, password_hash, role, full_name)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                "admin",
                "admin@hospital.com",
                generate_password_hash("admin123"),
                "admin",
                "System Administrator",
            ))
            conn.commit()
            log.info("Default admin user created  (admin / admin123)")
        cur.close()
        conn.close()
    except Exception as exc:
        log.warning(f"ensure_admin_user: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# ══  STARTUP  ═════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def startup():
    os.makedirs("models/saved_models", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    ensure_admin_user()

    if load_model_artifacts():
        log.info("✔  Sepsis model loaded and ready")
    else:
        log.warning(
            "⚠  No trained model found.\n"
            "   Run:  python train_model.py\n"
            "   Then restart the app."
        )

    try:
        nlp_processor.load_intent_classifier()
        log.info("✔  NLP / chatbot loaded")
    except Exception as exc:
        log.warning(f"Chatbot NLP not loaded: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# ══  ERROR HANDLERS  ══════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    log.error(f"500 error: {e}")
    return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    startup()
    app.run(debug=True, use_reloader=False, port=5000, host="0.0.0.0")