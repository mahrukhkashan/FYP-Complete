import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json
from datetime import datetime

class ModelEvaluator:
    """Evaluates model performance and generates reports"""
    
    def __init__(self, model, X_test, y_test, feature_names):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.results = {}
        
    def evaluate(self):
        """Run comprehensive evaluation"""
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        self.results = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'roc_curve': self._calculate_roc_curve(y_pred_proba),
            'pr_curve': self._calculate_pr_curve(y_pred_proba),
            'feature_importance': self._get_feature_importance(),
            'calibration': self._calculate_calibration(y_pred_proba)
        }
        
        return self.results
    
    def _calculate_roc_curve(self, y_pred_proba):
        """Calculate ROC curve data"""
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc)
        }
    
    def _calculate_pr_curve(self, y_pred_proba):
        """Calculate Precision-Recall curve data"""
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(pr_auc)
        }
    
    def _get_feature_importance(self):
        """Get feature importance from model"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            importance = np.zeros(len(self.feature_names))
        
        # Create sorted list of features by importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance.to_dict('records')
    
    def _calculate_calibration(self, y_pred_proba):
        """Calculate model calibration"""
        # Bin probabilities and calculate actual positive rate
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(y_pred_proba, bins) - 1
        
        calibration_data = []
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if mask.any():
                mean_pred = y_pred_proba[mask].mean()
                actual_pos_rate = self.y_test[mask].mean()
                count = mask.sum()
                
                calibration_data.append({
                    'bin_start': float(bins[i]),
                    'bin_end': float(bins[i+1]),
                    'mean_prediction': float(mean_pred),
                    'actual_positive_rate': float(actual_pos_rate),
                    'count': int(count)
                })
        
        return calibration_data
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': type(self.model).__name__,
            'dataset_size': {
                'test_samples': len(self.y_test),
                'positive_cases': int(self.y_test.sum()),
                'negative_cases': int(len(self.y_test) - self.y_test.sum()),
                'positive_rate': float(self.y_test.mean())
            },
            'performance_metrics': {
                'accuracy': float(self.results['classification_report']['accuracy']),
                'precision': float(self.results['classification_report']['weighted avg']['precision']),
                'recall': float(self.results['classification_report']['weighted avg']['recall']),
                'f1_score': float(self.results['classification_report']['weighted avg']['f1-score']),
                'roc_auc': self.results['roc_curve']['auc'],
                'pr_auc': self.results['pr_curve']['auc']
            },
            'class_metrics': self.results['classification_report'],
            'confusion_matrix': self.results['confusion_matrix'],
            'top_features': self.results['feature_importance'][:10],
            'calibration': self.results['calibration']
        }
        
        return report
    
    def plot_confusion_matrix(self):
        """Generate confusion matrix plot"""
        cm = np.array(self.results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Sepsis', 'Sepsis'],
                   yticklabels=['No Sepsis', 'Sepsis'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save to base64
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode('utf-8')
    
    def plot_roc_curve(self):
        """Generate ROC curve plot"""
        roc_data = self.results['roc_curve']
        
        plt.figure(figsize=(8, 6))
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                label=f'ROC curve (AUC = {roc_data["auc"]:.3f})',
                linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Save to base64
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode('utf-8')
    
    def plot_pr_curve(self):
        """Generate Precision-Recall curve plot"""
        pr_data = self.results['pr_curve']
        
        plt.figure(figsize=(8, 6))
        plt.plot(pr_data['recall'], pr_data['precision'], 
                label=f'PR curve (AUC = {pr_data["auc"]:.3f})',
                linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        # Save to base64
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode('utf-8')
    
    def plot_feature_importance(self, top_n=15):
        """Generate feature importance plot"""
        features = self.results['feature_importance'][:top_n]
        feature_names = [f['feature'] for f in features]
        importances = [f['importance'] for f in features]
        
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(feature_names))
        plt.barh(y_pos, importances, align='center', color='steelblue')
        plt.yticks(y_pos, feature_names)
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        
        # Save to base64
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode('utf-8')
    
    def plot_calibration_curve(self):
        """Generate calibration curve plot"""
        calibration_data = self.results['calibration']
        
        if not calibration_data:
            return None
        
        pred_probs = [d['mean_prediction'] for d in calibration_data]
        actual_probs = [d['actual_positive_rate'] for d in calibration_data]
        counts = [d['count'] for d in calibration_data]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(pred_probs, actual_probs, s=np.array(counts)*10, alpha=0.7)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Actual Positive Rate')
        plt.title('Calibration Curve (Bubble size = sample count)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save to base64
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode('utf-8')
    
    def generate_detailed_report(self, include_plots=True):
        """Generate detailed HTML report"""
        report = self.generate_report()
        
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 40px; }}
                .metric-card {{ 
                    background: #f8f9fa; 
                    border-radius: 8px; 
                    padding: 20px; 
                    margin: 10px; 
                    display: inline-block;
                    width: 200px;
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ color: #6c757d; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                img {{ max-width: 100%; height: auto; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Model Evaluation Report</h1>
            <div class="section">
                <h2>Summary</h2>
                <p>Report generated: {report['timestamp']}</p>
                <p>Model type: {report['model_type']}</p>
            </div>
            
            <div class="section">
                <h2>Dataset Information</h2>
                <p>Test samples: {report['dataset_size']['test_samples']}</p>
                <p>Positive cases: {report['dataset_size']['positive_cases']} ({report['dataset_size']['positive_rate']:.1%})</p>
                <p>Negative cases: {report['dataset_size']['negative_cases']}</p>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
        """
        
        # Add metric cards
        for metric, value in report['performance_metrics'].items():
            html_report += f"""
                <div class="metric-card">
                    <div class="metric-value">{value:.3f}</div>
                    <div class="metric-label">{metric.replace('_', ' ').title()}</div>
                </div>
            """
        
        html_report += """
            </div>
            
            <div class="section">
                <h2>Detailed Classification Report</h2>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                    </tr>
        """
        
        # Add classification report table
        for class_name, metrics in report['class_metrics'].items():
            if isinstance(metrics, dict):
                html_report += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{metrics['precision']:.3f}</td>
                        <td>{metrics['recall']:.3f}</td>
                        <td>{metrics['f1-score']:.3f}</td>
                        <td>{int(metrics['support'])}</td>
                    </tr>
                """
        
        html_report += """
                </table>
            </div>
        """
        
        # Add plots if requested
        if include_plots:
            html_report += """
            <div class="section">
                <h2>Visualizations</h2>
            """
            
            plots = [
                ('Confusion Matrix', self.plot_confusion_matrix()),
                ('ROC Curve', self.plot_roc_curve()),
                ('Precision-Recall Curve', self.plot_pr_curve()),
                ('Feature Importance', self.plot_feature_importance()),
                ('Calibration Curve', self.plot_calibration_curve())
            ]
            
            for plot_name, plot_data in plots:
                if plot_data:
                    html_report += f"""
                    <div class="plot">
                        <h3>{plot_name}</h3>
                        <img src="data:image/png;base64,{plot_data}" alt="{plot_name}">
                    </div>
                    """
        
        html_report += """
            </div>
            
            <div class="section">
                <h2>Top Features</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
        """
        
        # Add feature importance table
        for feature in report['top_features']:
            html_report += f"""
                <tr>
                    <td>{feature['feature']}</td>
                    <td>{feature['importance']:.4f}</td>
                </tr>
            """
        
        html_report += """
                </table>
            </div>
            
            <div class="section">
                <h2>Calibration Results</h2>
                <table>
                    <tr>
                        <th>Probability Bin</th>
                        <th>Mean Prediction</th>
                        <th>Actual Positive Rate</th>
                        <th>Sample Count</th>
                    </tr>
        """
        
        # Add calibration table
        for calib in report['calibration']:
            html_report += f"""
                <tr>
                    <td>{calib['bin_start']:.1f} - {calib['bin_end']:.1f}</td>
                    <td>{calib['mean_prediction']:.3f}</td>
                    <td>{calib['actual_positive_rate']:.3f}</td>
                    <td>{calib['count']}</td>
                </tr>
            """
        
        html_report += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html_report
    
    def save_report(self, filepath='models/saved_models/evaluation_report.json'):
        """Save evaluation report to file"""
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to {filepath}")
        return filepath