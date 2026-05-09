"""
API module for Flask web application.
"""

from api.app import app
from api.auth import auth_bp
from api.routes import api_bp

__all__ = ['app', 'auth_bp', 'api_bp']

# Version
__version__ = '1.0.0'