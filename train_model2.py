#!/usr/bin/env python3
"""
Script to train and save the sepsis prediction model for mimic_demo database
Optimized for small dataset (50 entries per table)
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

# Import your existing modules
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models.sepsis_predictor import SepsisPredictor
from explainability.shap_explainer import SHAPExplainer
from sklearn.model_selection import train_test_split

def train_and_save_model():
    """Main function to train and save the model for mimic_demo"""
    
    print("=" * 60)
    print("SEPSIS PREDICTION MODEL TRAINING - MIMIC_DEMO")
    print("Optimized for first 500 entries per table")
    print("=" * 60)
    
    # Create directories if they don't exist
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Step 1: Initialize DataLoader
    print("\n1. Initializing DataLoader for mimic_demo database...")
    
    try:
        data_loader = DataLoader()
        print("✓ Database connection successful")
        
        # Step 2: Load limited data (500 entries per table)
        print("\n2. Loading data (first 500 entries per table)...")
        
        # Load patient data
        patient_data = data_loader.load_patient_data(limit=500)
        
        if patient_data is None or patient_data.empty:
            print("⚠ No patient data loaded. Creating sample data...")
            patient_data = create_sample_data(500)
        else:
            print(f"✓ Loaded {len(patient_data)} patient records")
        
        # Display data info
        print(f"\nPatient data shape: {patient_data.shape}")
        print(f"Columns: {list(patient_data.columns)}")
        
        # Check for target column
        target_columns = ['has_sepsis', 'sepsis_label']
        target_col = None
        for col in target_columns:
            if col in patient_data.columns:
                target_col = col
                break
        
        if not target_col:
            print("No sepsis label found. Creating simulated labels...")
            patient_data = create_sepsis_labels(patient_data)
            target_col = 'has_sepsis'
        
        sepsis_count = patient_data[target_col].sum()
        sepsis_percent = (sepsis_count / len(patient_data)) * 100
        print(f"Sepsis cases: {sepsis_count} ({sepsis_percent:.1f}%)")
        
        # Step 3: Prepare features
        print("\n3. Preparing features...")
        
        # Separate features and target
        y = patient_data[target_col]
        
        # Drop non-feature columns
        cols_to_drop = [
            target_col,
            'subject_id', 'hadm_id', 'icustay_id', 
            'icd9_code', 'seq_num', 'charttime', 'chartdate',
            'text', 'description', 'category', 'measurement',
            'unit', 'flag', 'itemid', 'dob', 'admittime', 'dischtime'
        ]
        
        existing_cols_to_drop = [col for col in cols_to_drop if col in patient_data.columns]
        X = patient_data.drop(columns=existing_cols_to_drop)
        
        print(f"Dropped columns: {existing_cols_to_drop}")
        print(f"Features remaining: {list(X.columns)}")
        
        # Step 4: Handle missing values
        print("\n4. Handling missing values...")
        
        # Fill numeric columns with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().any():
                missing_count = X[col].isnull().sum()
                fill_value = X[col].median()
                X[col] = X[col].fillna(fill_value)
                print(f"  Filled {missing_count} missing values in '{col}' with {fill_value:.2f}")
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isnull().any():
                fill_value = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
                X[col] = X[col].fillna(fill_value)
        
        # Step 5: Feature engineering
        print("\n5. Feature engineering...")
        feature_engineer = FeatureEngineer()
        
        try:
            X_engineered = feature_engineer.engineer_features(X)
            print(f"Feature engineering completed")
            print(f"Original features: {X.shape[1]}, Engineered features: {X_engineered.shape[1]}")
        except Exception as e:
            print(f"Feature engineering failed: {e}")
            print("Using original features...")
            X_engineered = X.copy()
        
        # Encode categorical variables
        for col in X_engineered.select_dtypes(include=['object']).columns:
            X_engineered[col] = pd.factorize(X_engineered[col])[0]
        
        # Prepare final dataset
        X_final = X_engineered
        y_final = y
        
        # Ensure all features are numeric
        for col in X_final.columns:
            if not pd.api.types.is_numeric_dtype(X_final[col]):
                X_final[col] = pd.to_numeric(X_final[col], errors='coerce')
        
        # Fill any remaining NaN
        X_final = X_final.fillna(X_final.median())
        
        print(f"\nFinal dataset:")
        print(f"  Samples: {X_final.shape[0]}")
        print(f"  Features: {X_final.shape[1]}")
        print(f"  Sepsis cases: {y_final.sum()} ({y_final.mean()*100:.1f}%)")
        
        # Step 6: Train model
        print("\n6. Training model...")
        sepsis_predictor = SepsisPredictor()
        
        # Use smaller test size for small dataset
        test_size = 0.3 if len(X_final) > 30 else 0.2
        
        print(f"Using test size: {test_size}")
        results, X_test, y_test = sepsis_predictor.train(
            X_final, 
            y_final,
            test_size=test_size,
            random_state=42
        )
        
        # Print results
        print("\nModel Performance:")
        print("-" * 50)
        for model_name, result in results.items():
            metrics = result['metrics']
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Step 7: Save model and artifacts
        print("\n7. Saving model artifacts...")
        
        # Save model
        model_path = 'models/saved_models/sepsis_model.pkl'
        sepsis_predictor.save_model(model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Save feature names
        feature_names = list(X_final.columns)
        feature_names_path = 'models/saved_models/feature_names.json'
        with open(feature_names_path, 'w') as f:
            json.dump(feature_names, f)
        print(f"✓ Feature names saved to {feature_names_path}")
        
        # Save feature engineer
        feature_engineer_path = 'models/saved_models/feature_engineer.pkl'
        with open(feature_engineer_path, 'wb') as f:
            pickle.dump(feature_engineer, f)
        print(f"✓ Feature engineer saved to {feature_engineer_path}")
        
        # Step 8: Try SHAP explainer (optional for small dataset)
        if len(X_final) >= 20:
            print("\n8. Creating SHAP explainer...")
            try:
                X_train, _, y_train, _ = train_test_split(
                    X_final, y_final, test_size=test_size, random_state=42, stratify=y_final
                )
                
                shap_explainer = SHAPExplainer(sepsis_predictor.best_model, feature_names)
                shap_explainer.fit_explainer(X_train)
                
                shap_path = 'models/saved_models/shap_explainer.pkl'
                shap_explainer.save_explainer(shap_path)
                print(f"✓ SHAP explainer saved to {shap_path}")
                
            except Exception as e:
                print(f"⚠ SHAP explainer skipped: {e}")
        else:
            print("\n8. SKipping SHAP explainer (dataset too small)")
        
        # Close connection
        data_loader.close_connection()
        
    except Exception as e:
        print(f"\n✗ Error in training pipeline: {e}")
        print("Creating and saving a baseline model with sample data...")
        create_baseline_model()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)

def create_baseline_model():
    """Create a baseline model with sample data"""
    print("Creating baseline model with sample data...")
    
    # Create sample data
    patient_data = create_sample_data(100)
    
    # Prepare features
    X = patient_data.drop(columns=['subject_id', 'has_sepsis'])
    y = patient_data['has_sepsis']
    
    # Train a simple model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs('models/saved_models', exist_ok=True)
    
    model_path = 'models/saved_models/sepsis_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names
    feature_names = list(X.columns)
    feature_names_path = 'models/saved_models/feature_names.json'
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)
    
    # Save dummy feature engineer
    feature_engineer = FeatureEngineer()
    feature_engineer_path = 'models/saved_models/feature_engineer.pkl'
    with open(feature_engineer_path, 'wb') as f:
        pickle.dump(feature_engineer, f)
    
    print("✓ Baseline model created and saved")

def create_sepsis_labels(df):
    """Create simulated sepsis labels based on available features"""
    np.random.seed(42)
    
    print("Creating simulated sepsis labels...")
    
    sepsis_prob = np.zeros(len(df))
    
    # Use available features
    if 'age' in df.columns:
        sepsis_prob += (df['age'] > 65) * 0.2
    
    if 'heart_rate' in df.columns and pd.api.types.is_numeric_dtype(df['heart_rate']):
        sepsis_prob += (df['heart_rate'] > 100) * 0.2
    
    if 'temperature' in df.columns and pd.api.types.is_numeric_dtype(df['temperature']):
        sepsis_prob += (df['temperature'] > 37.8) * 0.2
    
    if 'respiratory_rate' in df.columns and pd.api.types.is_numeric_dtype(df['respiratory_rate']):
        sepsis_prob += (df['respiratory_rate'] > 20) * 0.15
    
    # Add random component
    sepsis_prob += np.random.random(len(df)) * 0.25
    sepsis_prob = np.clip(sepsis_prob, 0, 1)
    
    df['has_sepsis'] = (sepsis_prob > 0.5).astype(int)
    
    sepsis_count = df['has_sepsis'].sum()
    sepsis_percent = (sepsis_count / len(df)) * 100
    print(f"Created {sepsis_count} sepsis cases ({sepsis_percent:.1f}%)")
    
    return df

def create_sample_data(n_samples=500):
    """Create sample patient data"""
    np.random.seed(42)
    
    print(f"Generating {n_samples} sample records...")
    
    data = {
        'subject_id': range(1, n_samples + 1),
        'age': np.random.normal(60, 15, n_samples).clip(18, 95),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'heart_rate': np.random.normal(80, 12, n_samples).clip(50, 150),
        'temperature': np.random.normal(36.8, 0.8, n_samples).clip(35.5, 39.5),
        'respiratory_rate': np.random.normal(18, 4, n_samples).clip(12, 28),
        'systolic_bp': np.random.normal(125, 15, n_samples).clip(90, 180),
        'diastolic_bp': np.random.normal(80, 10, n_samples).clip(60, 110),
        'spo2': np.random.normal(96, 2, n_samples).clip(92, 100)
    }
    
    df = pd.DataFrame(data)
    
    # Simulate sepsis risk
    sepsis_prob = (
        (df['age'] > 70) * 0.3 +
        (df['heart_rate'] > 100) * 0.2 +
        (df['temperature'] > 37.8) * 0.2 +
        (df['respiratory_rate'] > 22) * 0.15 +
        np.random.random(n_samples) * 0.15
    )
    
    df['has_sepsis'] = (sepsis_prob > 0.5).astype(int)
    
    sepsis_count = df['has_sepsis'].sum()
    sepsis_percent = (sepsis_count / n_samples) * 100
    print(f"Sample data: {sepsis_count} sepsis cases ({sepsis_percent:.1f}%)")
    
    return df

if __name__ == '__main__':
    train_and_save_model()