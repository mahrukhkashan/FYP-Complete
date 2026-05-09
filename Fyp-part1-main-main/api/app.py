from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import pandas as pd
import json
from datetime import datetime
import os
from api.patient import patient_bp
from config.database_config import get_db_connection
from config.config import Config
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models.sepsis_predictor import SepsisPredictor
from explainability.shap_explainer import SHAPExplainer
from chatbot.nlp_processor import NLPProcessor
from chatbot.response_generator import ResponseGenerator
from utils.helpers import Helpers
from werkzeug.security import generate_password_hash, check_password_hash

# Blueprints
from api.auth import auth_bp
from api.routes import api_bp

# ------------------ APP INIT ------------------
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix="/auth")

app.register_blueprint(api_bp)
app.register_blueprint(patient_bp)


# ------------------ ML COMPONENTS ------------------
data_loader = DataLoader()
feature_engineer = FeatureEngineer()
sepsis_predictor = SepsisPredictor()
nlp_processor = NLPProcessor()
response_generator = ResponseGenerator()
helpers = Helpers()

model_trained = False
feature_names = []
shap_explainer = None

# ------------------ BASIC ROUTES ------------------
def create_admin_user():
    conn = get_db_connection()
    cur = conn.cursor()

    # Check if admin already exists
    cur.execute("SELECT id FROM users WHERE username = %s", ("admin",))
    if not cur.fetchone():
        # Insert admin user
        cur.execute("""
            INSERT INTO users (username, email, password_hash, role, full_name)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            "admin",
            "admin@example.com",
            generate_password_hash("admin123"),
            "admin",
            "Admin User"
        ))
        conn.commit()
        print("Admin user created successfully!")
    else:
        print("Admin user already exists.")

    cur.close()
    conn.close()

@app.route("/")
def signup_page():
    return render_template("index.html")


@app.route("/login")
def login_page():
    return render_template("login.html")



@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login_page"))

    role = session.get("role")

    if role == "admin":
        return render_template("admin_dashboard.html")
    elif role == "clinician":
        return render_template("clinician_dashboard.html")
    elif role == "attendant":
        return render_template("attendant_dashboard.html")
    else:
        return render_template("patientDashboard.html")

# ------------------ PREDICTION ------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        global model_trained, feature_names

        if not model_trained:
            sepsis_predictor.load_model("models/saved_models/sepsis_model.pkl")
            with open("models/saved_models/feature_names.json") as f:
                feature_names = json.load(f)
            model_trained = True

        patient_features = helpers.prepare_patient_features(data)

        fixed_features = {f: patient_features.get(f, 0) for f in feature_names}
        X = pd.DataFrame([fixed_features])

        result = sepsis_predictor.predict_single(
            X.iloc[0].to_dict(),
            feature_names,
            threshold=0.5
        )

        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "probability": float(result["probability"]),
            "risk_level": result["risk_level"],
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ CHATBOT ------------------

@app.route("/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.json
    message = data.get("message")

    intent = nlp_processor.extract_intent(message)
    entities = nlp_processor.extract_entities(message)

    response = response_generator.generate_response(
        intent=intent,
        entities=entities,
        context={},
        original_message=message
    )

    return jsonify(response)

# ------------------ EXPLANATION ------------------

@app.route("/explain", methods=["POST"])
def explain():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    global shap_explainer

    if not shap_explainer:
        shap_explainer = SHAPExplainer(sepsis_predictor.best_model, feature_names)
        shap_explainer.load_explainer("models/saved_models/shap_explainer.pkl")

    return jsonify({
        "success": True,
        "explanation": "SHAP explanation loaded successfully"
    })

# ------------------ TRAIN MODEL ------------------

@app.route("/train_model", methods=["POST"])
def train_model():
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    patient_data = data_loader.load_patient_data(limit=5000)
    vitals_data = data_loader.load_vitals_data(limit=10000)
    labs_data = data_loader.load_lab_data(limit=10000)

    merged = helpers.merge_patient_data(patient_data, vitals_data, labs_data)
    engineered = feature_engineer.engineer_features(merged)

    X = engineered.drop(["has_sepsis", "subject_id", "hadm_id"], axis=1, errors="ignore")
    y = engineered["has_sepsis"]

    global feature_names
    feature_names = list(X.columns)

    sepsis_predictor.train(X, y)
    sepsis_predictor.save_model()

    return jsonify({"success": True})

# ------------------ PATIENT DATA ------------------

@app.route("/patient/<int:patient_id>")
def get_patient(patient_id):
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    query = data_loader.sql.get_patient_full_data(patient_id)
    patient_data = data_loader.db.execute_query(query)

    if patient_data.empty:
        return jsonify({"error": "Patient not found"}), 404

    return jsonify(patient_data.iloc[0].to_dict())

# ------------------ APP START ------------------

if __name__ == "__main__":
    os.makedirs("models/saved_models", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)

    nlp_processor.load_intent_classifier()

    try:
        sepsis_predictor.load_model("models/saved_models/sepsis_model.pkl")
        with open("models/saved_models/feature_names.json") as f:
            feature_names = json.load(f)
        model_trained = True
        print("Model loaded successfully")
    except:
        print("No trained model found")
    create_admin_user()

    app.run(debug=True, use_reloader=False, port=5000)

