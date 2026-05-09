"""
Configuration module for application settings.
"""

from config.config import Config
from config.database_config import DatabaseConnection

__all__ = ['Config', 'DatabaseConnection']

# Version
__version__ = '1.0.0'