import polars as pl
import numpy as np
from typing import Dict, List
from pathlib import Path
from loguru import logger
import re
from PIL import Image
import io

class TaskDetector:
    """Intelligent ML task detection"""
    
    def __init__(self):
        self.text_indicators = ['text', 'description', 'comment', 'review', 'content', 'message']
        self.image_indicators = ['image', 'img', 'photo', 'picture', 'pic']
        self.time_indicators = ['date', 'time', 'timestamp', 'year', 'month', 'day']
        
    def detect_task(self, data_path: str) -> Dict:
        """Auto-detect ML task type from data"""
        logger.info(f"Detecting task type for {data_path}")
        
        df = pl.read_parquet(data_path)
        
        # Analyze columns
        column_analysis = self._analyze_columns(df)
        
        # Detect task type
        task_type = self._infer_task_type(df, column_analysis)
        
        # Identify target column
        target_column = self._identify_target_column(df, column_analysis, task_type)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(df, column_analysis, task_type)
        
        result = {
            'task_type': task_type,
            'ml_type': task_type,
            'target_column': target_column,
            'features': [col for col in df.columns if col != target_column],
            'column_analysis': column_analysis,
            'recommendations': recommendations
        }
        
        logger.info(f"Detected task: {task_type}, target: {target_column}")
        
        return result
    
    def _analyze_columns(self, df: pl.DataFrame) -> Dict:
        """Analyze column types and characteristics"""
        analysis = {}
        
        for col in df.columns:
            col_lower = col.lower()
            dtype = df[col].dtype
            unique_count = df[col].n_unique()
            null_count = df[col].null_count()
            total_count = len(df)
            
            col_info = {
                'dtype': str(dtype),
                'unique_count': unique_count,
                'null_percentage': (null_count / total_count) * 100,
                'cardinality': unique_count / total_count if total_count > 0 else 0
            }
            
            # Detect column purpose
            if any(ind in col_lower for ind in self.text_indicators):
                col_info['purpose'] = 'text'
            elif any(ind in col_lower for ind in self.image_indicators):
                col_info['purpose'] = 'image'
            elif any(ind in col_lower for ind in self.time_indicators):
                col_info['purpose'] = 'temporal'
            elif dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                col_info['purpose'] = 'numeric'
            elif dtype == pl.Utf8:
                # Check if categorical
                if col_info['cardinality'] < 0.05:  # Less than 5% unique
                    col_info['purpose'] = 'categorical'
                else:
                    # Check text length
                    sample_text = df[col].drop_nulls().head(10)
                    if len(sample_text) > 0:
                        avg_length = np.mean([len(str(x)) for x in sample_text])
                        col_info['purpose'] = 'text' if avg_length > 50 else 'categorical'
                    else:
                        col_info['purpose'] = 'categorical'
            else:
                col_info['purpose'] = 'unknown'
            
            # Statistical info for numeric
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                col_info['mean'] = float(df[col].mean()) if df[col].mean() else None
                col_info['std'] = float(df[col].std()) if df[col].std() else None
                col_info['min'] = float(df[col].min()) if df[col].min() else None
                col_info['max'] = float(df[col].max()) if df[col].max() else None
            
            analysis[col] = col_info
        
        return analysis
    
    def _infer_task_type(self, df: pl.DataFrame, column_analysis: Dict) -> str:
        """Infer ML task type from data characteristics"""
        
        # Check for text columns
        text_cols = [col for col, info in column_analysis.items() if info['purpose'] == 'text']
        if len(text_cols) > 0 and len(text_cols) / len(df.columns) > 0.3:
            return 'nlp'
        
        # Check for image columns
        image_cols = [col for col, info in column_analysis.items() if info['purpose'] == 'image']
        if len(image_cols) > 0:
            return 'computer_vision'
        
        # Check for temporal columns
        temporal_cols = [col for col, info in column_analysis.items() if info['purpose'] == 'temporal']
        if len(temporal_cols) > 0 and len(temporal_cols) / len(df.columns) > 0.2:
            return 'time_series'
        
        # Check for categorical target (classification)
        categorical_cols = [col for col, info in column_analysis.items() if info['purpose'] == 'categorical']
        if categorical_cols:
            # If last column is categorical with low cardinality, likely classification
            last_col = df.columns[-1]
            if last_col in categorical_cols and column_analysis[last_col]['cardinality'] < 0.1:
                return 'classification'
        
        # Check for continuous target (regression)
        numeric_cols = [col for col, info in column_analysis.items() if info['purpose'] == 'numeric']
        if numeric_cols:
            # If last column is continuous numeric, likely regression
            last_col = df.columns[-1]
            if last_col in numeric_cols and column_analysis[last_col]['cardinality'] > 0.1:
                return 'regression'
        
        # Default to classification
        return 'classification'
    
    def _identify_target_column(self, df: pl.DataFrame, column_analysis: Dict, task_type: str) -> str:
        """Identify the most likely target column"""
        
        # Common target column names
        target_names = ['target', 'label', 'class', 'outcome', 'result', 'y', 'prediction']
        
        # Check for explicit target names
        for col in df.columns:
            if any(name in col.lower() for name in target_names):
                return col
        
        # Default to last column
        return df.columns[-1]
    
    def _generate_recommendations(self, df: pl.DataFrame, column_analysis: Dict, task_type: str) -> List[str]:
        """Generate recommendations for ML pipeline"""
        recommendations = []
        
        # Data size recommendations
        if len(df) < 1000:
            recommendations.append("Small dataset - consider using simple models or data augmentation")
        elif len(df) > 1000000:
            recommendations.append("Large dataset - consider using GPU acceleration and batch training")
        
        # Missing data
        high_missing = [col for col, info in column_analysis.items() if info['null_percentage'] > 20]
        if high_missing:
            recommendations.append(f"High missing values in {len(high_missing)} columns - consider imputation or removal")
        
        # Imbalanced data
        if task_type == 'classification':
            target_col = self._identify_target_column(df, column_analysis, task_type)
            if target_col:
                try:
                    value_counts = df[target_col].value_counts()
                    if len(value_counts) > 0:
                        # Polars value_counts returns a DataFrame with 'count' column (not 'counts')
                        counts = value_counts.select('count').to_series()
                        max_count = counts.max()
                        min_count = counts.min()
                        if max_count / min_count > 3:
                            recommendations.append("Imbalanced classes detected - consider SMOTE or class weights")
                except:
                    pass
        
        # Feature count
        if len(df.columns) > 100:
            recommendations.append("High dimensionality - consider feature selection or PCA")
        
        # Model recommendations based on task
        if task_type == 'classification':
            recommendations.append("Suggested models: LightGBM, XGBoost, Random Forest")
        elif task_type == 'regression':
            recommendations.append("Suggested models: LightGBM, XGBoost, Linear Regression")
        elif task_type == 'nlp':
            recommendations.append("Suggested models: DistilBERT (GPU) or TF-IDF + Linear (CPU)")
        elif task_type == 'time_series':
            recommendations.append("Suggested models: LightGBM with lag features, ARIMA, Prophet")
        
        return recommendations