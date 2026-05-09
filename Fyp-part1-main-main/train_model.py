#!/usr/bin/env python3
"""
Script to train and save the sepsis prediction model
"""

import pandas as pd
import numpy as np
import json
import pickle
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models.sepsis_predictor import SepsisPredictor
from explainability.shap_explainer import SHAPExplainer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def train_and_save_model():
    """Main function to train and save the model"""
    
    print("=" * 60)
    print("SEPSIS PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    data_loader = DataLoader()
    
    try:
        # Load patient data
        patient_data = data_loader.load_patient_data(limit=10000)
        print(f"Loaded {len(patient_data)} patient records")
        
        # Load vitals data
        vitals_data = data_loader.load_vitals_data(limit=20000)
        print(f"Loaded {len(vitals_data)} vital sign records")
        
        # Load lab data
        labs_data = data_loader.load_lab_data(limit=15000)
        print(f"Loaded {len(labs_data)} lab result records")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nUsing sample data instead...")
        # Create sample data for testing
        patient_data = create_sample_data()
        vitals_data = pd.DataFrame()
        labs_data = pd.DataFrame()
    
    # Step 2: Merge and prepare data
    print("\n2. Preparing data...")
    feature_engineer = FeatureEngineer()
    
    # For this example, we'll create a merged dataset
    if not patient_data.empty:
        # Use patient data as base
        X = patient_data.copy()
    else:
        X = create_sample_features()
    
    # Add sepsis label (simulated for sample data)
    if 'has_sepsis' not in X.columns:
        np.random.seed(42)
        X['has_sepsis'] = np.random.choice([0, 1], size=len(X), p=[0.7, 0.3])
    
    # Engineer features
    X_engineered = feature_engineer.engineer_features(X)
    
    # Separate features and target
    y = X_engineered['has_sepsis']
    X_features = X_engineered.drop(['has_sepsis'], axis=1)

    # ========== ADD THIS CODE ==========
    print("\nConverting categorical columns for ML compatibility...")

    # Fix 'age_group' column specifically
    if 'age_group' in X_features.columns:
        print(f"  Found 'age_group' column with dtype: {X_features['age_group'].dtype}")
        
        # Convert from category to integer
        if X_features['age_group'].dtype.name == 'category':
            X_features['age_group'] = X_features['age_group'].cat.codes
            print("  Converted 'age_group' from category to integer codes")
        else:
            # Ensure it's integer
            X_features['age_group'] = X_features['age_group'].astype('int64')
            print("  Converted 'age_group' to int64")

    # Check for ANY other categorical columns
    categorical_cols = X_features.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        if col != 'age_group':  # Already handled
            X_features[col] = X_features[col].cat.codes
            print(f"  Converted '{col}' from category to integer codes")

    # Check for object/string columns
    object_cols = X_features.select_dtypes(include=['object']).columns
    for col in object_cols:
        X_features[col] = pd.factorize(X_features[col])[0]
        print(f"  Encoded '{col}' from object to integer")

    # Verify all columns are numeric
    print(f"\nFinal data types:")
    print(X_features.dtypes.value_counts())
    # ========== END ADDED CODE ==========
    
    # Handle any remaining missing values
    # X_features = X_features.fillna(X_features.mean())

    # Handle NaN values properly for different data types
    numeric_cols = X_features.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_features.select_dtypes(include=['object', 'category']).columns

    # Fill numeric columns with mean
    if len(numeric_cols) > 0:
        X_features[numeric_cols] = X_features[numeric_cols].fillna(X_features[numeric_cols].mean())

    # Fill categorical columns with mode
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            mode_val = X_features[col].mode()
            if len(mode_val) > 0:
                X_features[col] = X_features[col].fillna(mode_val[0])
            else:
                X_features[col] = X_features[col].fillna('Unknown')
    
    print(f"Final dataset shape: {X_features.shape}")
    print(f"Sepsis cases: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    
    # Step 3: Train model
    print("\n3. Training model...")
    sepsis_predictor = SepsisPredictor()
    
    results, X_test, y_test = sepsis_predictor.train(
        X_features, 
        y,
        test_size=0.2,
        random_state=42
    )
    
    # Print results
    print("\nModel Performance:")
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Step 4: Save model and artifacts
    print("\n4. Saving model artifacts...")
    
    # Save model
    sepsis_predictor.save_model('models/saved_models/sepsis_model.pkl')
    
    # Save feature names
    feature_names = list(X_features.columns)
    with open('models/saved_models/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    # Save feature engineer scaler and imputer
    with open('models/saved_models/feature_engineer.pkl', 'wb') as f:
        pickle.dump(feature_engineer, f)
    
    # Train and save SHAP explainer
    print("\n5. Training SHAP explainer...")
    X_train, _, y_train, _ = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    shap_explainer = SHAPExplainer(sepsis_predictor.best_model, feature_names)
    shap_explainer.fit_explainer(X_train)
    shap_explainer.save_explainer('models/saved_models/shap_explainer.pkl')
    
    # Step 5: Generate sample explanations
    print("\n6. Generating sample explanations...")
    
    # Get a sample prediction
    sample_idx = np.where(y_test == 1)[0][0] if any(y_test == 1) else 0
    sample_features = X_test.iloc[sample_idx:sample_idx+1]
    
    explanation = shap_explainer.explain_prediction(sample_features)
    
    print("\nSample Explanation:")
    print(f"Base value: {explanation['base_value']:.4f}")
    print("\nTop contributing features:")
    for i, effect in enumerate(explanation['feature_effects'][:5]):
        print(f"{i+1}. {effect['feature']}: {effect['shap_value']:.4f} ({effect['contribution']} risk)")
    
    # Close database connection
    data_loader.close_connection()
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the application: python run.py")
    print("2. Access the dashboard at: http://localhost:5000")
    print("3. Use username: 'demo' and password: 'demo' for testing")

def create_sample_data():
    """Create sample patient data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'subject_id': range(1, n_samples + 1),
        'age': np.random.randint(20, 80, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'admission_type': np.random.choice(['EMERGENCY', 'URGENT', 'ELECTIVE'], n_samples),
        'heart_rate': np.random.normal(85, 20, n_samples).clip(40, 180),
        'temperature': np.random.normal(37, 1, n_samples).clip(35, 41),
        'respiratory_rate': np.random.normal(18, 6, n_samples).clip(8, 40),
        'systolic_bp': np.random.normal(120, 25, n_samples).clip(80, 200),
        'diastolic_bp': np.random.normal(80, 15, n_samples).clip(50, 130),
        'wbc': np.random.normal(8, 4, n_samples).clip(2, 30),
        'lactate': np.random.exponential(1.5, n_samples).clip(0.5, 10),
        'creatinine': np.random.exponential(1.0, n_samples).clip(0.5, 8),
        'platelets': np.random.normal(250, 100, n_samples).clip(50, 500),
        'spo2': np.random.normal(96, 3, n_samples).clip(85, 100)
    }
    
    return pd.DataFrame(data)

def create_sample_features():
    """Create sample features for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create base features
    features = {
        'age': np.random.randint(20, 80, n_samples),
        'gender_M': np.random.choice([0, 1], n_samples),
        'admission_type_EMERGENCY': np.random.choice([0, 1], n_samples),
        'admission_type_URGENT': np.random.choice([0, 1], n_samples),
        'heart_rate': np.random.normal(85, 20, n_samples),
        'temperature': np.random.normal(37, 1, n_samples),
        'respiratory_rate': np.random.normal(18, 6, n_samples),
        'systolic_bp': np.random.normal(120, 25, n_samples),
        'diastolic_bp': np.random.normal(80, 15, n_samples),
        'wbc': np.random.normal(8, 4, n_samples),
        'lactate': np.random.exponential(1.5, n_samples),
        'creatinine': np.random.exponential(1.0, n_samples),
        'platelets': np.random.normal(250, 100, n_samples),
        'spo2': np.random.normal(96, 3, n_samples),
        'map': np.random.normal(93, 15, n_samples),
        'hr_abnormal': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'temp_abnormal': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'rr_abnormal': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'wbc_abnormal': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'lactate_abnormal': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'qsofa_score': np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'sirs_score': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'meets_sirs': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'age_group': np.random.choice([0, 1, 2, 3, 4], n_samples)
    }
    
    return pd.DataFrame(features)

if __name__ == '__main__':
    train_and_save_model()