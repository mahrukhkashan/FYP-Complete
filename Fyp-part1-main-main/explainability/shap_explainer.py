import shap
import numpy as np
import pandas as pd
import pickle

class SHAPExplainer:
    """SHAP-based model explainer"""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
    def fit_explainer(self, X):
        """Fit SHAP explainer to the model"""
        print("Fitting SHAP explainer...")
        
        # Check model type and use appropriate explainer
        model_type = type(self.model).__name__
        
        if any(tree_type in model_type for tree_type in ['RandomForest', 'XGB', 'GradientBoosting', 'DecisionTree']):
            # Tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            print(f"  Using TreeExplainer for {model_type}")
        else:
            # Linear models (Logistic Regression, etc.)
            self.explainer = shap.LinearExplainer(self.model, X)
            print(f"  Using LinearExplainer for {model_type}")
        
        # Alternative: KernelExplainer works for any model (but slower)
        # self.explainer = shap.KernelExplainer(self.model.predict_proba, X)
        
        return self.explainer
    
    def explain_prediction(self, X):
        """Generate SHAP explanation for a prediction"""
        if self.explainer is None:
            raise ValueError("Explainer must be fitted before making explanations")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # For classification with multiple classes
            shap_values = shap_values[1]  # Use class 1 (positive class)
        
        # Get base value
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]  # Use class 1
        
        # Calculate feature effects
        feature_effects = []
        for i, feature_name in enumerate(self.feature_names):
            shap_value = shap_values[0, i] if len(shap_values.shape) > 1 else shap_values[i]
            
            feature_effects.append({
                'feature': feature_name,
                'shap_value': float(shap_value),
                'contribution': 'increases' if shap_value > 0 else 'decreases'
            })
        
        # Sort by absolute importance
        feature_effects.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'base_value': float(base_value),
            'feature_effects': feature_effects,
            'prediction': float(self.model.predict_proba(X)[0, 1])
        }
    
    def save_explainer(self, path):
        """Save the SHAP explainer"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"SHAP explainer saved to {path}")
    
    @staticmethod
    def load_explainer(path):
        """Load a saved SHAP explainer"""
        with open(path, 'rb') as f:
            explainer = pickle.load(f)
        print(f"SHAP explainer loaded from {path}")
        return explainer