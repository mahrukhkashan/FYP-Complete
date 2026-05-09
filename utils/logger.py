import logging
import sys
from datetime import datetime
import os

class Logger:
    """Custom logger for the sepsis prediction system"""
    
    def __init__(self, name='sepsis_prediction'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        
        # File handler for all logs
        file_handler = logging.FileHandler(f'logs/sepsis_system_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        
        # File handler for errors only
        error_handler = logging.FileHandler(f'logs/errors_{datetime.now().strftime("%Y%m%d")}.log')
        error_handler.setLevel(logging.ERROR)
        error_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]')
        error_handler.setFormatter(error_format)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def get_logger(self):
        """Get the logger instance"""
        return self.logger
    
    def log_prediction(self, patient_id, prediction_result, confidence, features):
        """Log prediction events"""
        self.logger.info(f"PREDICTION - Patient: {patient_id}, Result: {prediction_result}, "
                        f"Confidence: {confidence:.3f}, Features: {features}")
    
    def log_chat_interaction(self, user_id, message, response, intent):
        """Log chatbot interactions"""
        self.logger.info(f"CHAT - User: {user_id}, Message: '{message[:50]}...', "
                        f"Intent: {intent}, Response: '{response[:50]}...'")
    
    def log_model_training(self, model_name, metrics, duration):
        """Log model training events"""
        self.logger.info(f"MODEL_TRAINING - Model: {model_name}, Metrics: {metrics}, "
                        f"Duration: {duration:.2f}s")
    
    def log_error(self, error_type, error_message, context=None):
        """Log error events"""
        error_context = f"Context: {context}" if context else ""
        self.logger.error(f"ERROR - Type: {error_type}, Message: {error_message}, {error_context}")
    
    def log_system_event(self, event_type, description):
        """Log system events"""
        self.logger.info(f"SYSTEM - Event: {event_type}, Description: {description}")
    
    def log_user_action(self, user_id, action, details):
        """Log user actions"""
        self.logger.info(f"USER_ACTION - User: {user_id}, Action: {action}, Details: {details}")

# Create global logger instance
logger = Logger().get_logger()

def get_logger():
    """Get the global logger instance"""
    return logger