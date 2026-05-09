from flask import Blueprint, jsonify, request, session # pyright: ignore[reportMissingImports]
from datetime import datetime, timedelta
import random
from utils.logger import get_logger
from flask import render_template # pyright: ignore[reportMissingImports]

logger = get_logger()

api_bp = Blueprint('api', __name__)

# Sample data for dashboard
sample_patients = [
    {'id': 1001, 'name': 'John Smith', 'age': 45, 'risk_level': 'High Risk', 'last_prediction': '2 hours ago'},
    {'id': 1002, 'name': 'Mary Johnson', 'age': 62, 'risk_level': 'Medium Risk', 'last_prediction': '4 hours ago'},
    {'id': 1003, 'name': 'Robert Brown', 'age': 38, 'risk_level': 'Low Risk', 'last_prediction': '1 day ago'},
    {'id': 1004, 'name': 'Sarah Davis', 'age': 71, 'risk_level': 'High Risk', 'last_prediction': '30 minutes ago'},
    {'id': 1005, 'name': 'Michael Wilson', 'age': 56, 'risk_level': 'Medium Risk', 'last_prediction': '6 hours ago'},
    {'id': 1006, 'name': 'Emily Taylor', 'age': 29, 'risk_level': 'Low Risk', 'last_prediction': '2 days ago'},
    {'id': 1007, 'name': 'David Martinez', 'age': 48, 'risk_level': 'High Risk', 'last_prediction': '1 hour ago'},
    {'id': 1008, 'name': 'Jennifer Anderson', 'age': 33, 'risk_level': 'Medium Risk', 'last_prediction': '8 hours ago'}
]

@api_bp.route('/')
def home():
    return render_template('index.html')

@api_bp.route('/login')
def login():
    return render_template('login.html')

@api_bp.route('/forgot-password')
def forgot_password():
    return render_template('forgotpassword.html')

@api_bp.route('/dashboard_stats', methods=['GET'])
def dashboard_stats():
    """Get dashboard statistics"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Sample statistics
    stats = {
        'total_predictions': random.randint(500, 1000),
        'high_risk_cases': random.randint(50, 100),
        'medium_risk_cases': random.randint(150, 200),
        'low_risk_cases': random.randint(300, 400),
        'model_accuracy': round(random.uniform(85, 95), 1),
        'total_patients': len(sample_patients),
        'avg_response_time': round(random.uniform(1.5, 3.0), 2),
        'system_uptime': '99.8%'
    }
    
    return jsonify({
        'success': True,
        'stats': stats
    })

@api_bp.route('/recent_patients', methods=['GET'])
def recent_patients():
    """Get recent patients"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Return sample patients
    return jsonify({
        'success': True,
        'patients': sample_patients
    })

@api_bp.route('/patient/<int:patient_id>', methods=['GET'])
def get_patient_details(patient_id):
    """Get patient details"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Find patient in sample data
    patient = None
    for p in sample_patients:
        if p['id'] == patient_id:
            patient = p
            break
    
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    
    # Add more details
    patient_details = {
        **patient,
        'gender': random.choice(['Male', 'Female']),
        'blood_type': random.choice(['A+', 'B+', 'O+', 'AB+']),
        'admission_date': (datetime.now() - timedelta(days=random.randint(1, 10))).strftime('%Y-%m-%d'),
        'primary_diagnosis': random.choice(['Pneumonia', 'UTI', 'Cellulitis', 'Appendicitis']),
        'room_number': f'ICU-{random.randint(1, 20)}',
        'attending_physician': 'Dr. Smith'
    }
    
    # Add vital signs
    vital_signs = {
        'heart_rate': random.randint(70, 120),
        'temperature': round(random.uniform(36.5, 39.5), 1),
        'respiratory_rate': random.randint(12, 25),
        'blood_pressure': f"{random.randint(100, 160)}/{random.randint(60, 100)}",
        'oxygen_saturation': random.randint(90, 100),
        'last_updated': '10 minutes ago'
    }
    
    # Add lab results
    lab_results = {
        'wbc': round(random.uniform(4.0, 15.0), 1),
        'lactate': round(random.uniform(1.0, 4.0), 1),
        'creatinine': round(random.uniform(0.7, 2.5), 2),
        'platelets': random.randint(150, 400),
        'bilirubin': round(random.uniform(0.5, 3.0), 1),
        'last_updated': '2 hours ago'
    }
    
    # Add prediction history
    prediction_history = []
    for i in range(3):
        prediction_history.append({
            'timestamp': (datetime.now() - timedelta(hours=i*6)).strftime('%Y-%m-%d %H:%M'),
            'risk_level': random.choice(['Low Risk', 'Medium Risk', 'High Risk']),
            'probability': round(random.uniform(0.2, 0.9), 3),
            'model_confidence': round(random.uniform(0.85, 0.95), 2)
        })
    
    return jsonify({
        'success': True,
        'patient': patient_details,
        'vital_signs': vital_signs,
        'lab_results': lab_results,
        'prediction_history': prediction_history
    })

@api_bp.route('/alerts', methods=['GET'])
def get_alerts():
    """Get system alerts"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Sample alerts
    alerts = [
        {
            'id': 1,
            'patient_id': 1004,
            'patient_name': 'Sarah Davis',
            'alert_type': 'Critical Vital',
            'severity': 'High',
            'message': 'Blood pressure critically low: 85/45 mmHg',
            'timestamp': '15 minutes ago',
            'status': 'Active'
        },
        {
            'id': 2,
            'patient_id': 1001,
            'patient_name': 'John Smith',
            'alert_type': 'High Risk Prediction',
            'severity': 'High',
            'message': 'Sepsis risk increased to 92%',
            'timestamp': '45 minutes ago',
            'status': 'Active'
        },
        {
            'id': 3,
            'patient_id': 1007,
            'patient_name': 'David Martinez',
            'alert_type': 'Lab Abnormality',
            'severity': 'Medium',
            'message': 'Lactate level elevated: 3.8 mmol/L',
            'timestamp': '2 hours ago',
            'status': 'Acknowledged'
        }
    ]
    
    return jsonify({
        'success': True,
        'alerts': alerts
    })

@api_bp.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    logger.info(f"Alert {alert_id} acknowledged by user {session.get('username')}")
    
    return jsonify({
        'success': True,
        'message': f'Alert {alert_id} acknowledged'
    })

@api_bp.route('/predictions/history', methods=['GET'])
def prediction_history():
    """Get prediction history"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Sample prediction history
    history = []
    for i in range(10):
        history.append({
            'id': i + 1,
            'patient_id': random.choice([1001, 1002, 1003, 1004]),
            'patient_name': random.choice(['John Smith', 'Mary Johnson', 'Robert Brown', 'Sarah Davis']),
            'timestamp': (datetime.now() - timedelta(hours=i*2)).strftime('%Y-%m-%d %H:%M'),
            'risk_level': random.choice(['Low Risk', 'Medium Risk', 'High Risk']),
            'probability': round(random.uniform(0.1, 0.95), 3),
            'model_used': random.choice(['Random Forest', 'XGBoost', 'Ensemble']),
            'was_correct': random.choice([True, False, None])
        })
    
    return jsonify({
        'success': True,
        'history': history
    })

@api_bp.route('/model/performance', methods=['GET'])
def model_performance():
    """Get model performance metrics"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Sample performance metrics
    performance = {
        'accuracy': round(random.uniform(0.85, 0.95), 3),
        'precision': round(random.uniform(0.82, 0.92), 3),
        'recall': round(random.uniform(0.83, 0.93), 3),
        'f1_score': round(random.uniform(0.84, 0.94), 3),
        'roc_auc': round(random.uniform(0.88, 0.98), 3),
        'training_date': '2024-01-15',
        'training_samples': 5000,
        'feature_count': 25,
        'model_type': 'Random Forest'
    }
    
    # Feature importance
    feature_importance = [
        {'feature': 'Lactate Level', 'importance': 0.25},
        {'feature': 'Heart Rate', 'importance': 0.18},
        {'feature': 'Temperature', 'importance': 0.15},
        {'feature': 'WBC Count', 'importance': 0.12},
        {'feature': 'Respiratory Rate', 'importance': 0.10},
        {'feature': 'Age', 'importance': 0.08},
        {'feature': 'Systolic BP', 'importance': 0.06},
        {'feature': 'Platelets', 'importance': 0.04},
        {'feature': 'Creatinine', 'importance': 0.02}
    ]
    
    return jsonify({
        'success': True,
        'performance': performance,
        'feature_importance': feature_importance
    })

@api_bp.route('/system/status', methods=['GET'])
def system_status():
    """Get system status"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    status = {
        'api_status': 'Online',
        'database_status': 'Connected',
        'model_status': 'Loaded',
        'chatbot_status': 'Active',
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'uptime': '99.5%',
        'response_time': round(random.uniform(0.5, 2.0), 2),
        'active_users': random.randint(5, 20)
    }
    
    return jsonify({
        'success': True,
        'status': status
    })

@api_bp.route('/export/predictions', methods=['GET'])
def export_predictions():
    """Export predictions as CSV"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # In production, this would generate and return a CSV file
    # For now, return sample data
    sample_csv = """patient_id,patient_name,timestamp,risk_level,probability,model_used
1001,John Smith,2024-01-15 10:30,High Risk,0.92,Random Forest
1002,Mary Johnson,2024-01-15 11:45,Medium Risk,0.65,XGBoost
1003,Robert Brown,2024-01-15 12:15,Low Risk,0.28,Ensemble
1004,Sarah Davis,2024-01-15 14:20,High Risk,0.88,Random Forest"""
    
    return jsonify({
        'success': True,
        'csv_data': sample_csv,
        'filename': f'predictions_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    })