import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt')
    nltk.download('stopwords')

class FeatureEngineer:
    """Handles feature engineering for sepsis prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.stop_words = set(stopwords.words('english'))
        
    def engineer_features(self, df, text_data=None):
        """Engineer features from raw data"""
        print("Engineering features...")
        
        # 1. Handle demographics
        df = self._process_demographics(df)
        
        # 2. Process vital signs
        df = self._process_vitals(df)
        
        # 3. Process lab results
        df = self._process_labs(df)
        
        # 4. Calculate derived features
        df = self._calculate_derived_features(df)
        
        # 5. Handle missing values
        df = self._handle_missing_values(df)
        
        # 6. Add text features if available
        if text_data is not None and not text_data.empty:
            text_features = self._extract_text_features(text_data)
            df = pd.concat([df, text_features], axis=1)
        
        return df
    
    def _process_demographics(self, df):
        """Process demographic features"""
        # Encode categorical variables
        categorical_cols = ['gender', 'admission_type', 'ethnicity', 'first_careunit']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str).fillna('Unknown'))
                self.label_encoders[col] = le
        
        # Create age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                     bins=[0, 30, 50, 65, 80, 100], 
                                     labels=[0, 1, 2, 3, 4])
        
        return df
    
    def _process_vitals(self, df):
        """Process vital signs"""
        vital_cols = ['heart_rate', 'systolic_bp', 'diastolic_bp', 
                     'temperature', 'respiratory_rate', 'spo2']
        
        # Calculate MAP (Mean Arterial Pressure)
        if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
            df['map'] = df['diastolic_bp'] + (df['systolic_bp'] - df['diastolic_bp']) / 3
        
        # Flag abnormal values
        if 'heart_rate' in df.columns:
            df['hr_abnormal'] = ((df['heart_rate'] < 60) | (df['heart_rate'] > 100)).astype(int)
        
        if 'temperature' in df.columns:
            df['temp_abnormal'] = ((df['temperature'] < 36) | (df['temperature'] > 38)).astype(int)
        
        if 'respiratory_rate' in df.columns:
            df['rr_abnormal'] = ((df['respiratory_rate'] < 12) | (df['respiratory_rate'] > 20)).astype(int)
        
        return df
    
    def _process_labs(self, df):
        """Process laboratory results"""
        lab_cols = ['wbc', 'lactate', 'creatinine', 'platelets', 'bilirubin']
        
        # Calculate SOFA score components
        if 'platelets' in df.columns:
            df['platelet_score'] = np.select([
                df['platelets'] >= 150,
                (df['platelets'] >= 100) & (df['platelets'] < 150),
                (df['platelets'] >= 50) & (df['platelets'] < 100),
                (df['platelets'] >= 20) & (df['platelets'] < 50),
                df['platelets'] < 20
            ], [0, 1, 2, 3, 4], default=0)
        
        if 'creatinine' in df.columns:
            df['creatinine_score'] = np.select([
                df['creatinine'] < 1.2,
                (df['creatinine'] >= 1.2) & (df['creatinine'] <= 1.9),
                (df['creatinine'] >= 2.0) & (df['creatinine'] <= 3.4),
                (df['creatinine'] >= 3.5) & (df['creatinine'] <= 4.9),
                df['creatinine'] > 5.0
            ], [0, 1, 2, 3, 4], default=0)
        
        if 'bilirubin' in df.columns:
            df['bilirubin_score'] = np.select([
                df['bilirubin'] < 1.2,
                (df['bilirubin'] >= 1.2) & (df['bilirubin'] <= 1.9),
                (df['bilirubin'] >= 2.0) & (df['bilirubin'] <= 5.9),
                (df['bilirubin'] >= 6.0) & (df['bilirubin'] <= 11.9),
                df['bilirubin'] >= 12.0
            ], [0, 1, 2, 3, 4], default=0)
        
        # Flag abnormal values
        if 'wbc' in df.columns:
            df['wbc_abnormal'] = ((df['wbc'] < 4) | (df['wbc'] > 11)).astype(int)
        
        if 'lactate' in df.columns:
            df['lactate_abnormal'] = (df['lactate'] > 2).astype(int)
        
        return df
    
    def _calculate_derived_features(self, df):
        """Calculate derived clinical scores"""
        # Calculate quick SOFA (qSOFA)
        qsofa_features = []
        if 'respiratory_rate' in df.columns:
            qsofa_features.append((df['respiratory_rate'] >= 22).astype(int))
        if 'systolic_bp' in df.columns:
            qsofa_features.append((df['systolic_bp'] <= 100).astype(int))
        if 'mental_status' in df.columns:  # This would need GCS data
            qsofa_features.append((df['mental_status'] <= 13).astype(int))
        
        if qsofa_features:
            df['qsofa_score'] = sum(qsofa_features)
        
        # Calculate SIRS criteria
        sirs_features = []
        if 'temperature' in df.columns:
            sirs_features.append(((df['temperature'] >= 38) | (df['temperature'] <= 36)).astype(int))
        if 'heart_rate' in df.columns:
            sirs_features.append((df['heart_rate'] >= 90).astype(int))
        if 'respiratory_rate' in df.columns:
            sirs_features.append((df['respiratory_rate'] >= 20).astype(int))
        if 'wbc' in df.columns:
            sirs_features.append(((df['wbc'] >= 12) | (df['wbc'] <= 4)).astype(int))
        
        if sirs_features:
            df['sirs_score'] = sum(sirs_features)
            df['meets_sirs'] = (df['sirs_score'] >= 2).astype(int)
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values"""
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # Fill categorical columns with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def _extract_text_features(self, text_data):
        """Extract features from clinical notes"""
        # Clean text
        text_data['cleaned_text'] = text_data['text'].apply(self._clean_text)
        
        # Get TF-IDF features
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_data['cleaned_text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                               columns=self.tfidf_vectorizer.get_feature_names_out())
        
        # Add sepsis-related keywords
        sepsis_keywords = ['sepsis', 'infection', 'fever', 'hypotension', 
                          'tachycardia', 'tachypnea', 'wbc', 'lactate']
        
        for keyword in sepsis_keywords:
            tfidf_df[f'contains_{keyword}'] = text_data['cleaned_text'].str.contains(keyword, case=False).astype(int)
        
        return tfidf_df
    
    def _clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def scale_features(self, df):
        """Scale features using StandardScaler"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df, self.scaler
    
    def get_feature_importance_template(self):
        """Return template for feature importance explanation"""
        return {
            'vital_signs': ['heart_rate', 'systolic_bp', 'temperature', 'respiratory_rate'],
            'lab_results': ['lactate', 'wbc', 'creatinine', 'platelets'],
            'demographics': ['age', 'gender', 'admission_type'],
            'clinical_scores': ['qsofa_score', 'sirs_score', 'meets_sirs'],
            'derived_features': ['hr_abnormal', 'temp_abnormal', 'rr_abnormal', 'lactate_abnormal']
        }