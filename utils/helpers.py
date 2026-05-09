import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
import re

class Helpers:
    """Utility helper functions"""
    
    @staticmethod
    def prepare_patient_features(data):
        """Prepare patient features for prediction"""
        features = {
            'age': float(data.get('age', 50)),
            'gender_M': 1 if data.get('gender', '').upper() == 'M' else 0,
            'admission_type_EMERGENCY': 1 if data.get('admission_type', '').upper() == 'EMERGENCY' else 0,
            'admission_type_URGENT': 1 if data.get('admission_type', '').upper() == 'URGENT' else 0,
            'heart_rate': float(data.get('heart_rate', 80)),
            'temperature': float(data.get('temperature', 37)),
            'respiratory_rate': float(data.get('respiratory_rate', 18)),
            'systolic_bp': float(data.get('systolic_bp', 120)),
            'diastolic_bp': float(data.get('diastolic_bp', 80)),
            'wbc': float(data.get('wbc', 8)),
            'lactate': float(data.get('lactate', 1.5)),
            'creatinine': float(data.get('creatinine', 1.0)),
            'platelets': float(data.get('platelets', 250)),
            'spo2': float(data.get('spo2', 96)),
            'map': float(data.get('diastolic_bp', 80)) + (float(data.get('systolic_bp', 120)) - float(data.get('diastolic_bp', 80))) / 3,
            'hr_abnormal': 1 if float(data.get('heart_rate', 80)) < 60 or float(data.get('heart_rate', 80)) > 100 else 0,
            'temp_abnormal': 1 if float(data.get('temperature', 37)) < 36 or float(data.get('temperature', 37)) > 38 else 0,
            'rr_abnormal': 1 if float(data.get('respiratory_rate', 18)) < 12 or float(data.get('respiratory_rate', 18)) > 20 else 0,
            'wbc_abnormal': 1 if float(data.get('wbc', 8)) < 4 or float(data.get('wbc', 8)) > 11 else 0,
            'lactate_abnormal': 1 if float(data.get('lactate', 1.5)) > 2 else 0,
            'qsofa_score': 0,
            'sirs_score': 0,
            'meets_sirs': 0,
            'age_group': 2  # Default middle age group
        }
        
        # Calculate qSOFA score
        qsofa = 0
        if float(data.get('respiratory_rate', 18)) >= 22:
            qsofa += 1
        if float(data.get('systolic_bp', 120)) <= 100:
            qsofa += 1
        features['qsofa_score'] = qsofa
        
        # Calculate SIRS criteria
        sirs = 0
        if float(data.get('temperature', 37)) >= 38 or float(data.get('temperature', 37)) <= 36:
            sirs += 1
        if float(data.get('heart_rate', 80)) >= 90:
            sirs += 1
        if float(data.get('respiratory_rate', 18)) >= 20:
            sirs += 1
        if float(data.get('wbc', 8)) >= 12 or float(data.get('wbc', 8)) <= 4:
            sirs += 1
        
        features['sirs_score'] = sirs
        features['meets_sirs'] = 1 if sirs >= 2 else 0
        
        # Calculate age group
        age = float(data.get('age', 50))
        if age <= 30:
            features['age_group'] = 0
        elif age <= 50:
            features['age_group'] = 1
        elif age <= 65:
            features['age_group'] = 2
        elif age <= 80:
            features['age_group'] = 3
        else:
            features['age_group'] = 4
        
        return features
    
    @staticmethod
    def merge_patient_data(patient_data, vitals_data, labs_data):
        """Merge different data sources"""
        merged = patient_data.copy()
        
        if not vitals_data.empty:
            # Aggregate vitals by patient
            vitals_agg = vitals_data.groupby('subject_id').agg({
                'heart_rate': 'mean',
                'systolic_bp': 'mean',
                'diastolic_bp': 'mean',
                'temperature': 'mean',
                'respiratory_rate': 'mean',
                'spo2': 'mean'
            }).reset_index()
            
            merged = pd.merge(merged, vitals_agg, on='subject_id', how='left')
        
        if not labs_data.empty:
            # Aggregate labs by patient
            labs_agg = labs_data.groupby('subject_id').agg({
                'wbc': 'mean',
                'lactate': 'mean',
                'creatinine': 'mean',
                'platelets': 'mean'
            }).reset_index()
            
            merged = pd.merge(merged, labs_agg, on='subject_id', how='left')
        
        return merged
    
    @staticmethod
    def validate_patient_data(data):
        """Validate patient data"""
        errors = []
        
        required_fields = ['age', 'heart_rate', 'temperature', 'respiratory_rate']
        for field in required_fields:
            if field not in data or data[field] == '':
                errors.append(f"Missing required field: {field}")
        
        # Validate numeric ranges
        if 'age' in data:
            try:
                age = float(data['age'])
                if age < 0 or age > 120:
                    errors.append("Age must be between 0 and 120")
            except:
                errors.append("Age must be a number")
        
        if 'heart_rate' in data:
            try:
                hr = float(data['heart_rate'])
                if hr < 30 or hr > 250:
                    errors.append("Heart rate must be between 30 and 250 bpm")
            except:
                errors.append("Heart rate must be a number")
        
        if 'temperature' in data:
            try:
                temp = float(data['temperature'])
                if temp < 30 or temp > 45:
                    errors.append("Temperature must be between 30 and 45°C")
            except:
                errors.append("Temperature must be a number")
        
        return errors
    
    @staticmethod
    def calculate_clinical_scores(features):
        """Calculate clinical scores from features"""
        scores = {}
        
        # Calculate SOFA score components
        if 'platelets' in features:
            platelets = features['platelets']
            if platelets >= 150:
                scores['platelet_score'] = 0
            elif platelets >= 100:
                scores['platelet_score'] = 1
            elif platelets >= 50:
                scores['platelet_score'] = 2
            elif platelets >= 20:
                scores['platelet_score'] = 3
            else:
                scores['platelet_score'] = 4
        
        if 'creatinine' in features:
            creatinine = features['creatinine']
            if creatinine < 1.2:
                scores['creatinine_score'] = 0
            elif creatinine <= 1.9:
                scores['creatinine_score'] = 1
            elif creatinine <= 3.4:
                scores['creatinine_score'] = 2
            elif creatinine <= 4.9:
                scores['creatinine_score'] = 3
            else:
                scores['creatinine_score'] = 4
        
        # Calculate total qSOFA
        qsofa = 0
        if features.get('respiratory_rate', 0) >= 22:
            qsofa += 1
        if features.get('systolic_bp', 0) <= 100:
            qsofa += 1
        scores['qsofa_total'] = qsofa
        
        return scores
    
    @staticmethod
    def generate_patient_summary(features, prediction):
        """Generate patient summary for display"""
        summary = {
            'demographics': {
                'age': features.get('age', 'Unknown'),
                'gender': 'Male' if features.get('gender_M', 0) == 1 else 'Female',
                'admission_type': 'Emergency' if features.get('admission_type_EMERGENCY', 0) == 1 else 'Urgent' if features.get('admission_type_URGENT', 0) == 1 else 'Elective'
            },
            'vitals': {
                'heart_rate': f"{features.get('heart_rate', 0):.0f} bpm",
                'temperature': f"{features.get('temperature', 0):.1f}°C",
                'respiratory_rate': f"{features.get('respiratory_rate', 0):.0f} breaths/min",
                'blood_pressure': f"{features.get('systolic_bp', 0):.0f}/{features.get('diastolic_bp', 0):.0f} mmHg",
                'oxygen_saturation': f"{features.get('spo2', 0):.0f}%"
            },
            'labs': {
                'wbc': f"{features.get('wbc', 0):.1f} ×10⁹/L",
                'lactate': f"{features.get('lactate', 0):.1f} mmol/L",
                'creatinine': f"{features.get('creatinine', 0):.2f} mg/dL",
                'platelets': f"{features.get('platelets', 0):.0f} ×10⁹/L"
            },
            'prediction': {
                'risk_level': prediction.get('risk_level', 'Unknown'),
                'probability': f"{prediction.get('probability', 0) * 100:.1f}%",
                'confidence': 'High' if prediction.get('probability', 0) > 0.8 else 'Medium' if prediction.get('probability', 0) > 0.6 else 'Low'
            }
        }
        
        return summary
    
    @staticmethod
    def format_explanation_for_display(explanation):
        """Format explanation for web display"""
        if not explanation:
            return "No explanation available"
        
        formatted = {
            'risk_factors': [],
            'protective_factors': [],
            'recommendations': []
        }
        
        # Parse SHAP explanation
        if 'feature_effects' in explanation:
            for effect in explanation['feature_effects'][:5]:
                factor = {
                    'feature': effect['feature'].replace('_', ' ').title(),
                    'impact': abs(effect['shap_value']),
                    'direction': effect['contribution'],
                    'description': Helpers._get_feature_description(effect['feature'])
                }
                
                if effect['shap_value'] > 0:
                    formatted['risk_factors'].append(factor)
                else:
                    formatted['protective_factors'].append(factor)
        
        # Add clinical recommendations
        formatted['recommendations'] = Helpers._generate_recommendations(formatted['risk_factors'])
        
        return formatted
    
    @staticmethod
    def _get_feature_description(feature_name):
        """Get human-readable description for features"""
        descriptions = {
            'lactate': 'Blood lactate level - high levels indicate tissue hypoxia',
            'heart_rate': 'Heart rate - tachycardia can indicate systemic inflammation',
            'temperature': 'Body temperature - fever is a sign of infection',
            'wbc': 'White blood cell count - elevated in infection',
            'respiratory_rate': 'Respiratory rate - tachypnea indicates respiratory distress',
            'age': 'Patient age - older patients have higher risk',
            'systolic_bp': 'Systolic blood pressure - hypotension indicates shock',
            'spo2': 'Oxygen saturation - low levels indicate respiratory failure',
            'creatinine': 'Creatinine level - high levels indicate kidney dysfunction',
            'platelets': 'Platelet count - low levels indicate coagulation issues'
        }
        
        return descriptions.get(feature_name, f"{feature_name.replace('_', ' ').title()}")
    
    @staticmethod
    def _generate_recommendations(risk_factors):
        """Generate clinical recommendations based on risk factors"""
        recommendations = []
        
        for factor in risk_factors:
            feature = factor['feature'].lower()
            
            if 'lactate' in feature:
                recommendations.append('Consider blood gas analysis and fluid resuscitation')
            elif 'heart' in feature:
                recommendations.append('Monitor cardiac rhythm and consider ECG')
            elif 'temperature' in feature:
                recommendations.append('Check for infection source and consider antipyretics')
            elif 'wbc' in feature or 'white blood' in feature:
                recommendations.append('Order blood cultures and consider antibiotics')
            elif 'respiratory' in feature:
                recommendations.append('Assess respiratory status and consider oxygen therapy')
            elif 'blood pressure' in feature or 'bp' in feature:
                recommendations.append('Monitor blood pressure closely and consider vasopressors')
            elif 'oxygen' in feature or 'spo2' in feature:
                recommendations.append('Check oxygen saturation and consider supplemental oxygen')
        
        # Add general recommendations
        recommendations.append('Consider sepsis bundle implementation')
        recommendations.append('Reassess patient in 1-2 hours')
        recommendations.append('Consult infectious disease specialist if available')
        
        return list(set(recommendations))[:5]  # Return top 5 unique recommendations
    
    @staticmethod
    def hash_password(password):
        """Hash password for storage"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def check_password(password, hashed):
        """Check password against hash"""
        return Helpers.hash_password(password) == hashed
    
    @staticmethod
    def generate_patient_id():
        """Generate a unique patient ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_part = hashlib.md5(str(np.random.random()).encode()).hexdigest()[:6]
        return f"PT{timestamp}{random_part}".upper()
    
    @staticmethod
    def sanitize_text(text):
        """Sanitize text input"""
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Limit length
        if len(text) > 10000:
            text = text[:10000] + "..."
        
        return text.strip()