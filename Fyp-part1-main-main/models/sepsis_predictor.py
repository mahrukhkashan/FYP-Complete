import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class SepsisPredictor:
    """Main class for sepsis prediction using multiple ML models"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced'
            ),
            'xgboost': XGBClassifier(
                n_estimators=100, 
                random_state=42, 
                use_label_encoder=False, 
                eval_metric='logloss',
                enable_categorical=True,  # CRITICAL: Add this
                tree_method='hist'  # Optional: Better for Windows
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, 
                class_weight='balanced', 
                max_iter=1000
            )
        }
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.scaler = None
        self.encoder = None
        
    def train(self, X, y, test_size=0.2, random_state=42):
        """Train multiple models and select the best one"""
        print("Training sepsis prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        results = {}
        best_score = 0
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Check if this is the best model (using F1-score as primary metric)
            if metrics['f1_score'] > best_score:
                best_score = metrics['f1_score']
                self.best_model = model
                self.best_model_name = model_name
                self.feature_importance = self._get_feature_importance(model, X_train.columns)
        
        print(f"\nBest model: {self.best_model_name} with F1-score: {best_score:.4f}")
        return results, X_test, y_test
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate performance metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def _get_feature_importance(self, model, feature_names):
        """Get feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            importance = np.zeros(len(feature_names))
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def predict(self, X, threshold=0.5):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Get probabilities
        probabilities = self.best_model.predict_proba(X)[:, 1]
        
        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
    
    def predict_single(self, patient_features, feature_names, threshold=0.5):
        """Predict sepsis risk for a single patient"""
        # Ensure features are in correct order
        patient_df = pd.DataFrame([patient_features], columns=feature_names)
        
        # Make prediction
        prediction, probability = self.predict(patient_df, threshold)
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'risk_level': self._get_risk_level(float(probability[0]))
        }
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def save_model(self, path='models/saved_models/sepsis_model.pkl'):
        """Save the trained model"""
        if self.best_model is not None:
            joblib.dump({
                'model': self.best_model,
                'model_name': self.best_model_name,
                'feature_importance': self.feature_importance
            }, path)
            print(f"Model saved to {path}")
    
    def load_model(self, path='models/saved_models/sepsis_model.pkl'):
        """Load a trained model"""
        loaded = joblib.load(path)
        self.best_model = loaded['model']
        self.best_model_name = loaded['model_name']
        self.feature_importance = loaded['feature_importance']
        print(f"Model loaded from {path}")
    
    def get_model_summary(self):
        """Get summary of the trained model"""
        if self.best_model is None:
            return "Model not trained yet"
        
        summary = {
            'model_type': self.best_model_name,
            'model_parameters': self.best_model.get_params(),
            'feature_count': len(self.feature_importance) if self.feature_importance is not None else 0,
            'top_features': self.feature_importance.head(10).to_dict('records') if self.feature_importance is not None else []
        }
        
        return summary