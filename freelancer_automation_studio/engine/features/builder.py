import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
from pathlib import Path
from loguru import logger
import joblib

class FeatureBuilder:
    """Feature engineering and train/test splitting"""
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        
    def build_features(
        self,
        data_path: str,
        task_info: Dict,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[Any, Any, Any, Any]:
        """Build features and split data"""
        logger.info("Building features...")
        
        df = pl.read_parquet(data_path)
        
        # Identify target and features
        target_col = task_info.get('target_column')
        task_type = task_info.get('ml_type', 'classification')
        
        if not target_col:
            # For unsupervised, return all features
            X = df.to_pandas()
            return self._prepare_features(X, task_type), None, None, None
        
        # Separate features and target
        if target_col in df.columns:
            y = df[target_col].to_pandas()
            X = df.drop(target_col).to_pandas()
        else:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Feature engineering based on task type
        if task_type == 'time_series':
            X = self._build_time_series_features(X, df)
            # Use time-series split
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            # Standard train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if task_type == 'classification' and len(np.unique(y)) < len(y) * 0.5 else None
            )
        
        # Prepare features
        X_train = self._prepare_features(X_train, task_type, fit=True)
        X_test = self._prepare_features(X_test, task_type, fit=False)
        
        # Encode labels for classification
        if task_type == 'classification':
            y_train, y_test = self._encode_labels(y_train, y_test)
        
        logger.info(f"Features built: {X_train.shape[1]} features, {X_train.shape[0]} train samples")
        
        return X_train, X_test, y_train, y_test
    
    def _prepare_features(self, X: Any, task_type: str, fit: bool = False) -> np.ndarray:
        """Prepare features for ML"""
        # Handle different data types
        if isinstance(X, pl.DataFrame):
            X = X.to_pandas()
        
        # Encode categorical columns
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen labels
                        X[col] = X[col].astype(str).map(
                            lambda x: self.label_encoders[col].transform([x])[0]
                            if x in self.label_encoders[col].classes_
                            else -1
                        )
                    else:
                        X[col] = -1
        
        # Convert to numpy
        X = X.values
        
        # Scale features for certain task types
        if task_type in ['regression', 'classification']:
            if fit:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            elif self.scaler:
                X = self.scaler.transform(X)
        
        return X
    
    def _build_time_series_features(self, X: Any, df: pl.DataFrame) -> Any:
        """Build lag features for time series"""
        logger.info("Building time series features...")
        
        # Create lag features
        lags = [1, 2, 3, 7, 14, 30]
        
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                for lag in lags:
                    X[f'{col}_lag_{lag}'] = X[col].shift(lag)
                
                # Rolling statistics
                X[f'{col}_rolling_mean_7'] = X[col].rolling(window=7).mean()
                X[f'{col}_rolling_std_7'] = X[col].rolling(window=7).std()
        
        # Drop NaN created by lags
        X = X.dropna()
        
        return X
    
    def _encode_labels(self, y_train: Any, y_test: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Encode target labels for classification"""
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        self.label_encoders['__target__'] = le
        
        return y_train_encoded, y_test_encoded
    
    def save_preprocessing(self, path: Path):
        """Save preprocessing objects"""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)