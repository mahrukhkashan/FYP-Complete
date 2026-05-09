import pandas as pd
import numpy as np
from data.sql_queries import SQLQueries
from config.database_config import DatabaseConnection
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Loads data from MIMIC-III database"""
    
    def __init__(self):
        self.db = DatabaseConnection()
        self.db.connect()
        self.sql = SQLQueries()
    
    def load_patient_data(self, subject_id=None, limit=1000):
        """Load patient data with sepsis indicators"""
        print("Loading patient data...")
        
        # Load admissions data
        admissions_query = self.sql.get_admissions_query(limit)
        admissions_df = self.db.execute_query(admissions_query)
        
        # Load patient demographics
        patients_query = self.sql.get_patients_query(limit)
        patients_df = self.db.execute_query(patients_query)
        
        # Load ICU stays
        icustays_query = self.sql.get_icustays_query(limit)
        icustays_df = self.db.execute_query(icustays_query)
        
        # Load diagnoses (to identify sepsis)
        diagnoses_query = self.sql.get_diagnoses_query(limit)
        diagnoses_df = self.db.execute_query(diagnoses_query)
        
        # Merge data
        merged_df = self._merge_data(admissions_df, patients_df, icustays_df, diagnoses_df)
        
        # Add sepsis label
        merged_df = self._add_sepsis_label(merged_df)
        
        return merged_df
    
    def _merge_data(self, admissions, patients, icustays, diagnoses):
        """Merge different data sources"""
        # Merge admissions with patients
        merged = pd.merge(admissions, patients, on='subject_id', how='left')
        
        # Merge with ICU stays
        merged = pd.merge(merged, icustays, on=['subject_id', 'hadm_id'], how='left')
        
        # Add diagnoses
        sepsis_diagnoses = diagnoses[diagnoses['icd9_code'].str.startswith(('038', '785.52', '995.91', '995.92'), na=False)]
        sepsis_diagnoses['has_sepsis'] = 1
        sepsis_diagnoses = sepsis_diagnoses[['subject_id', 'hadm_id', 'has_sepsis']].drop_duplicates()
        
        merged = pd.merge(merged, sepsis_diagnoses, on=['subject_id', 'hadm_id'], how='left')
        
        return merged
    
    def _add_sepsis_label(self, df):
        """Add sepsis label based on ICD-9 codes"""
        df['has_sepsis'] = df['has_sepsis'].fillna(0).astype(int)
        return df
    
    def load_vitals_data(self, subject_ids=None, limit=10000):
        """Load vital signs data from chartevents"""
        print("Loading vital signs data...")
        
        if subject_ids:
            vitals_query = self.sql.get_vitals_query_with_subjects(subject_ids, limit)
        else:
            vitals_query = self.sql.get_vitals_query(limit)
        
        vitals_df = self.db.execute_query(vitals_query)
        return vitals_df
    
    def load_lab_data(self, subject_ids=None, limit=10000):
        """Load laboratory results"""
        print("Loading laboratory data...")
        
        if subject_ids:
            labs_query = self.sql.get_labs_query_with_subjects(subject_ids, limit)
        else:
            labs_query = self.sql.get_labs_query(limit)
        
        labs_df = self.db.execute_query(labs_query)
        return labs_df
    
    def load_notes_data(self, subject_ids=None, limit=1000):
        """Load clinical notes"""
        print("Loading clinical notes...")
        
        if subject_ids:
            notes_query = self.sql.get_notes_query_with_subjects(subject_ids, limit)
        else:
            notes_query = self.sql.get_notes_query(limit)
        
        notes_df = self.db.execute_query(notes_query)
        return notes_df
    
    def close_connection(self):
        """Close database connection"""
        self.db.close()