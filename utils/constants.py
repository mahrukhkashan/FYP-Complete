"""
Constants for the sepsis prediction system
"""

# Clinical thresholds
VITAL_THRESHOLDS = {
    'HEART_RATE': {'min': 60, 'max': 100, 'critical_min': 40, 'critical_max': 140},
    'TEMPERATURE': {'min': 36.0, 'max': 38.0, 'critical_min': 35.0, 'critical_max': 40.0},
    'RESPIRATORY_RATE': {'min': 12, 'max': 20, 'critical_min': 8, 'critical_max': 30},
    'SYSTOLIC_BP': {'min': 90, 'max': 140, 'critical_min': 70, 'critical_max': 200},
    'DIASTOLIC_BP': {'min': 60, 'max': 90, 'critical_min': 40, 'critical_max': 120},
    'OXYGEN_SATURATION': {'min': 94, 'max': 100, 'critical_min': 90, 'critical_max': 100},
    'WBC': {'min': 4.0, 'max': 11.0, 'critical_min': 2.0, 'critical_max': 30.0},
    'LACTATE': {'min': 0.5, 'max': 2.0, 'critical_min': 0.5, 'critical_max': 10.0},
    'CREATININE': {'min': 0.6, 'max': 1.2, 'critical_min': 0.5, 'critical_max': 8.0},
    'PLATELETS': {'min': 150, 'max': 450, 'critical_min': 50, 'critical_max': 1000}
}

# Sepsis criteria
SEPSIS_CRITERIA = {
    'SIRS': {
        'temperature': {'min': 38.0, 'max': 36.0},
        'heart_rate': {'min': 90},
        'respiratory_rate': {'min': 20},
        'wbc': {'min': 12.0, 'max': 4.0}
    },
    'QSOFA': {
        'respiratory_rate': {'min': 22},
        'systolic_bp': {'max': 100},
        'altered_mental_status': True
    },
    'SOFA': {
        'respiration': {'pao2_fio2': {'max': 400}},
        'coagulation': {'platelets': {'max': 150}},
        'liver': {'bilirubin': {'min': 1.2}},
        'cardiovascular': {'map': {'max': 70}},
        'cns': {'glasgow_coma_scale': {'max': 15}},
        'renal': {'creatinine': {'min': 1.2}}
    }
}

# Risk levels
RISK_LEVELS = {
    'LOW': {'min': 0.0, 'max': 0.3, 'color': '#28a745', 'action': 'Monitor'},
    'MEDIUM': {'min': 0.3, 'max': 0.7, 'color': '#ffc107', 'action': 'Close monitoring'},
    'HIGH': {'min': 0.7, 'max': 1.0, 'color': '#dc3545', 'action': 'Immediate intervention'}
}

# Model parameters
MODEL_PARAMS = {
    'RANDOM_FOREST': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced'
    },
    'XGBOOST': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    'LOGISTIC_REGRESSION': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'class_weight': 'balanced'
    }
}

# Feature categories
FEATURE_CATEGORIES = {
    'DEMOGRAPHICS': ['age', 'gender', 'admission_type', 'ethnicity'],
    'VITAL_SIGNS': ['heart_rate', 'temperature', 'respiratory_rate', 'systolic_bp', 'diastolic_bp', 'spo2'],
    'LABORATORY': ['wbc', 'lactate', 'creatinine', 'platelets', 'bilirubin', 'crp'],
    'CLINICAL_SCORES': ['qsofa_score', 'sirs_score', 'meets_sirs'],
    'DERIVED_FEATURES': ['map', 'hr_abnormal', 'temp_abnormal', 'rr_abnormal', 'wbc_abnormal', 'lactate_abnormal']
}

# Alert types
ALERT_TYPES = {
    'CRITICAL_VITAL': 'Critical vital sign',
    'HIGH_RISK_PREDICTION': 'High sepsis risk prediction',
    'LAB_ABNORMALITY': 'Abnormal lab result',
    'TREND_WORSENING': 'Deteriorating trend',
    'SYSTEM_ERROR': 'System error'
}

# User roles
USER_ROLES = {
    'ADMIN': 'admin',
    'CLINICIAN': 'clinician',
    'NURSE': 'nurse',
    'PATIENT': 'patient',
    'RESEARCHER': 'researcher'
}

# Chatbot intents
CHATBOT_INTENTS = {
    'GREETING': 'greeting',
    'SEPSIS_RISK': 'sepsis_risk',
    'EXPLANATION': 'explanation',
    'SYMPTOMS': 'symptoms',
    'PREVENTION': 'prevention',
    'TREATMENT': 'treatment',
    'HELP': 'help',
    'GOODBYE': 'goodbye'
}

# API endpoints
API_ENDPOINTS = {
    'PREDICT': '/predict',
    'CHAT': '/chat',
    'EXPLAIN': '/explain',
    'PATIENT': '/patient',
    'TRAIN': '/train_model',
    'DASHBOARD': '/dashboard_stats',
    'ALERTS': '/alerts'
}

# Error messages
ERROR_MESSAGES = {
    'DATABASE_CONNECTION': 'Database connection failed',
    'MODEL_NOT_TRAINED': 'Model not trained. Please train the model first.',
    'INVALID_INPUT': 'Invalid input data',
    'PATIENT_NOT_FOUND': 'Patient not found',
    'UNAUTHORIZED': 'Unauthorized access',
    'INTERNAL_ERROR': 'Internal server error'
}

# Success messages
SUCCESS_MESSAGES = {
    'PREDICTION_SUCCESS': 'Prediction completed successfully',
    'MODEL_TRAINED': 'Model trained successfully',
    'DATA_SAVED': 'Data saved successfully',
    'USER_CREATED': 'User created successfully',
    'ACTION_COMPLETED': 'Action completed successfully'
}

# Time constants (in seconds)
TIME_CONSTANTS = {
    'PREDICTION_TIMEOUT': 5,
    'CHAT_RESPONSE_TIMEOUT': 3,
    'MODEL_TRAINING_TIMEOUT': 300,
    'SESSION_TIMEOUT': 3600,
    'DATA_REFRESH_INTERVAL': 60
}

# File paths
FILE_PATHS = {
    'MODEL_SAVE': 'models/saved_models/sepsis_model.pkl',
    'SHAP_EXPLAINER': 'models/saved_models/shap_explainer.pkl',
    'FEATURE_NAMES': 'models/saved_models/feature_names.json',
    'SCALER': 'models/saved_models/scaler.pkl',
    'ENCODER': 'models/saved_models/encoder.pkl',
    'CONFIG': 'config/config.py',
    'LOGS_DIR': 'logs',
    'UPLOADS_DIR': 'uploads'
}

# Database tables
DATABASE_TABLES = {
    'USERS': 'users',
    'PATIENTS': 'patients',
    'PREDICTIONS': 'predictions',
    'VITALS': 'patient_vitals',
    'LABS': 'lab_results',
    'CHAT': 'chat_conversations',
    'ALERTS': 'alerts'
}

# Performance metrics targets
PERFORMANCE_TARGETS = {
    'ACCURACY': 0.85,
    'PRECISION': 0.80,
    'RECALL': 0.85,
    'F1_SCORE': 0.82,
    'ROC_AUC': 0.90,
    'RESPONSE_TIME': 3.0
}