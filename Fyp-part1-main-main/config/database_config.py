import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from config.config import Config


def get_db_connection():
    return psycopg2.connect(Config.SQLALCHEMY_DATABASE_URI)

class DatabaseConnection:
    """Handles database connection and queries"""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                host="localhost",
                database="mimiciii",
                user="postgres",
                password="Fozia786",  # Change this to your password
                port="5432"
            )
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            print("Database connection established successfully")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    def execute_query(self, query, params=None):
        """Execute a SQL query and return results"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            # For SELECT queries, fetch results
            if query.strip().upper().startswith('SELECT'):
                result = self.cursor.fetchall()
                return pd.DataFrame(result) if result else pd.DataFrame()
            else:
                self.connection.commit()
                return True
        except Exception as e:
            print(f"Query execution failed: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
    