import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from loguru import logger
import io
import base64
from pathlib import Path

class ModelExplainer:
    """Model explainability using SHAP"""
    
    def __init__(self, model: Any, task_type: str):
        self.model = model
        self.task_type = task_type
        self.explainer = None
        
    def explain(self, X_test: Any, y_test: Any = None, max_samples: int = 100) -> Dict:
        """Generate comprehensive explanations"""
        logger.info("Generating SHAP explanations...")
        
        # Limit samples for performance
        X_sample = X_test[:max_samples] if len(X_test) > max_samples else X_test
        
        explanations = {}
        
        try:
            # Feature importance
            explanations['feature_importance'] = self._get_feature_importance()
            
            # SHAP values
            explanations['shap_values'] = self._compute_shap_values(X_sample)
            
            # SHAP summary plot
            explanations['shap_summary'] = self._generate_shap_plot(X_sample)
            
            # Error analysis (if y_test provided)
            if y_test is not None:
                explanations['error_analysis'] = self._analyze_errors(X_test, y_test)
            
            logger.info("Explanations generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
            explanations['error'] = str(e)
        
        return explanations
    
    def _get_feature_importance(self) -> list:
        """Extract feature importance from model"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                
                # Create feature names
                feature_names = [f'feature_{i}' for i in range(len(importances))]
                
                # Sort by importance
                indices = np.argsort(importances)[::-1]
                
                return [
                    {'feature': feature_names[i], 'importance': float(importances[i])}
                    for i in indices
                ]
            else:
                logger.warning("Model does not have feature_importances_ attribute")
                return []
        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}")
            return []
    
    def _compute_shap_values(self, X_sample: Any) -> Any:
        """Compute SHAP values"""
        try:
            # Use TreeExplainer for tree-based models
            if hasattr(self.model, 'predict_proba'):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Use KernelExplainer as fallback
                self.explainer = shap.KernelExplainer(self.model.predict, X_sample[:50])
            
            shap_values = self.explainer.shap_values(X_sample)
            
            return shap_values
        
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return None
    
    def _generate_shap_plot(self, X_sample: Any) -> str:
        """Generate SHAP summary plot"""
        try:
            if self.explainer is None:
                return None
            
            shap_values = self.explainer.shap_values(X_sample)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            
            if isinstance(shap_values, list):
                # Multi-class classification
                shap.summary_plot(shap_values[0], X_sample, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, show=False)
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            plt.close()
            
            # Convert to base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return f"data:image/png;base64,{img_base64}"
        
        except Exception as e:
            logger.error(f"Error generating SHAP plot: {e}")
            return None
    
    def _analyze_errors(self, X_test: Any, y_test: Any) -> Dict:
        """Analyze prediction errors"""
        try:
            y_pred = self.model.predict(X_test)
            
            if self.task_type in ['classification', 'nlp']:
                # Classification metrics
                errors = y_pred != y_test
                error_rate = errors.sum() / len(y_test)
                
                # Confusion by class
                unique_classes = np.unique(y_test)
                class_errors = {}
                for cls in unique_classes:
                    cls_mask = y_test == cls
                    cls_error_rate = errors[cls_mask].sum() / cls_mask.sum() if cls_mask.sum() > 0 else 0
                    class_errors[f'class_{cls}'] = float(cls_error_rate)
                
                return {
                    'overall_error_rate': float(error_rate),
                    'class_error_rates': class_errors,
                    'total_errors': int(errors.sum())
                }
            
            else:  # Regression
                errors = np.abs(y_pred - y_test)
                
                return {
                    'mean_absolute_error': float(errors.mean()),
                    'max_error': float(errors.max()),
                    'min_error': float(errors.min()),
                    'error_std': float(errors.std())
                }
        
        except Exception as e:
            logger.error(f"Error analyzing errors: {e}")
            return {'error': str(e)}