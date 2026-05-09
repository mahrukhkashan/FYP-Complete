-- Create database for sepsis prediction system
CREATE DATABASE sepsis_prediction_system;

-- Connect to the database
\c sepsis_prediction_system;

-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'patient',
    full_name VARCHAR(100),
    specialization VARCHAR(100),
    hospital_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Patients table (extends users)
CREATE TABLE patients (
    user_id INTEGER PRIMARY KEY REFERENCES users(id),
    date_of_birth DATE,
    gender VARCHAR(10),
    blood_type VARCHAR(5),
    height_cm DECIMAL(5,2),
    weight_kg DECIMAL(5,2),
    emergency_contact VARCHAR(100),
    medical_history TEXT,
    allergies TEXT,
    current_medications TEXT
);

-- Predictions table
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES users(id),
    clinician_id INTEGER REFERENCES users(id),
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    risk_probability DECIMAL(5,4),
    risk_level VARCHAR(20),
    prediction_result BOOLEAN,
    confidence_score DECIMAL(5,4),
    model_version VARCHAR(50),
    input_features JSONB,
    explanation JSONB,
    is_correct BOOLEAN,
    notes TEXT
);

-- Patient vitals table
CREATE TABLE patient_vitals (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES users(id),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    heart_rate INTEGER,
    systolic_bp INTEGER,
    diastolic_bp INTEGER,
    temperature DECIMAL(4,2),
    respiratory_rate INTEGER,
    oxygen_saturation DECIMAL(4,2),
    pain_level INTEGER,
    recorded_by INTEGER REFERENCES users(id)
);

-- Lab results table
CREATE TABLE lab_results (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES users(id),
    test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    test_type VARCHAR(100),
    wbc DECIMAL(6,2),
    lactate DECIMAL(5,2),
    creatinine DECIMAL(5,2),
    platelets INTEGER,
    bilirubin DECIMAL(5,2),
    crp DECIMAL(6,2),
    procalcitonin DECIMAL(6,2),
    result_status VARCHAR(20),
    lab_notes TEXT
);

-- Chat conversations table
CREATE TABLE chat_conversations (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES users(id),
    clinician_id INTEGER REFERENCES users(id),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    conversation_summary TEXT
);

-- Chat messages table
CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES chat_conversations(id),
    sender_id INTEGER REFERENCES users(id),
    message_text TEXT,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_type VARCHAR(20), -- 'text', 'image', 'prediction', 'explanation'
    metadata JSONB
);

-- Model performance logs
CREATE TABLE model_logs (
    id SERIAL PRIMARY KEY,
    log_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),
    training_data_size INTEGER,
    test_data_size INTEGER,
    accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    roc_auc DECIMAL(5,4),
    training_duration_seconds INTEGER,
    hyperparameters JSONB
);

-- Alerts table
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES users(id),
    alert_type VARCHAR(50),
    severity VARCHAR(20),
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    acknowledged_by INTEGER REFERENCES users(id),
    resolution_status VARCHAR(20) DEFAULT 'pending'
);

-- Create indexes for performance
CREATE INDEX idx_predictions_patient_id ON predictions(patient_id);
CREATE INDEX idx_predictions_timestamp ON predictions(prediction_timestamp);
CREATE INDEX idx_vitals_patient_id ON patient_vitals(patient_id);
CREATE INDEX idx_vitals_recorded_at ON patient_vitals(recorded_at);
CREATE INDEX idx_chat_messages_conversation_id ON chat_messages(conversation_id);
CREATE INDEX idx_chat_messages_sender_id ON chat_messages(sender_id);
CREATE INDEX idx_alerts_patient_id ON alerts(patient_id);
CREATE INDEX idx_alerts_status ON alerts(resolution_status);

-- Insert sample users
INSERT INTO users (username, email, password_hash, role, full_name, specialization) VALUES
('dr_smith', 'dr.smith@hospital.com', 'hashed_password_1', 'clinician', 'Dr. John Smith', 'Critical Care'),
('nurse_jones', 'nurse.jones@hospital.com', 'hashed_password_2', 'clinician', 'Nurse Sarah Jones', 'ICU Nursing'),
('patient_doe', 'john.doe@email.com', 'hashed_password_3', 'patient', 'John Doe', NULL);

-- Insert sample patient
INSERT INTO patients (user_id, date_of_birth, gender, blood_type, height_cm, weight_kg) VALUES
(3, '1980-05-15', 'Male', 'O+', 180.5, 85.2);

-- Create a view for patient dashboard
CREATE VIEW patient_dashboard_view AS
SELECT 
    u.id as patient_id,
    u.full_name,
    p.date_of_birth,
    p.gender,
    p.blood_type,
    (SELECT risk_level FROM predictions WHERE patient_id = u.id ORDER BY prediction_timestamp DESC LIMIT 1) as latest_risk_level,
    (SELECT risk_probability FROM predictions WHERE patient_id = u.id ORDER BY prediction_timestamp DESC LIMIT 1) as latest_risk_probability,
    (SELECT COUNT(*) FROM predictions WHERE patient_id = u.id) as total_predictions,
    (SELECT recorded_at FROM patient_vitals WHERE patient_id = u.id ORDER BY recorded_at DESC LIMIT 1) as last_vital_check
FROM users u
LEFT JOIN patients p ON u.id = p.user_id
WHERE u.role = 'patient';