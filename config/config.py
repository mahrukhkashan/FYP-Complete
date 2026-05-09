import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:Fozia786@localhost:5432/mimiciii'  # Use your correct password
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Model paths
    MODEL_DIR = 'models/saved_models'
    SCALER_PATH = 'models/saved_models/scaler.pkl'
    ENCODER_PATH = 'models/saved_models/label_encoder.pkl'
    
    # File upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Chatbot settings
    CHATBOT_NAME = "Sepsis Assistant"
    DEFAULT_RESPONSES = {
        'greeting': "Hello! I'm your sepsis prediction assistant. How can I help you today?",
        'fallback': "I'm sorry, I didn't understand that. Could you rephrase?",
        'help': "I can help you with: 1) Checking sepsis risk 2) Explaining predictions 3) Preventive measures"
    }