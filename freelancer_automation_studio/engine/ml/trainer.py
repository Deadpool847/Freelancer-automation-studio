import numpy as np
from typing import Dict, Tuple, Any, Optional, Callable
from loguru import logger
import optuna
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Tabular models
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

# Text models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

class ModelTrainer:
    """Unified model trainer for all task types"""
    
    def __init__(self, task_type: str, use_gpu: bool = False, n_jobs: int = 4):
        self.task_type = task_type
        self.use_gpu = use_gpu
        self.n_jobs = n_jobs
        self.best_model = None
        
    def train(
        self,
        X_train: Any,
        X_test: Any,
        y_train: Any,
        y_test: Any,
        optuna_trials: int = 20,
        cv_folds: int = 5,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Any, Dict]:
        """Train model with hyperparameter optimization"""
        logger.info(f"Training {self.task_type} model...")
        
        if self.task_type in ['classification', 'nlp']:
            model, metrics = self._train_classification(X_train, X_test, y_train, y_test, optuna_trials, cv_folds, progress_callback)
        elif self.task_type in ['regression', 'time_series']:
            model, metrics = self._train_regression(X_train, X_test, y_train, y_test, optuna_trials, cv_folds, progress_callback)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        self.best_model = model
        return model, metrics
    
    def _train_classification(self, X_train, X_test, y_train, y_test, optuna_trials, cv_folds, progress_callback):
        """Train classification model"""
        
        # Define objective for Optuna
        def objective(trial):
            model_name = trial.suggest_categorical('model', ['lightgbm', 'xgboost', 'random_forest'])
            
            if model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'verbosity': -1,
                    'n_jobs': self.n_jobs
                }
                model = LGBMClassifier(**params)
            
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
                    'verbosity': 0,
                    'n_jobs': self.n_jobs
                }
                model = XGBClassifier(**params)
            
            else:  # random_forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'n_jobs': self.n_jobs
                }
                model = RandomForestClassifier(**params)
            
            # Cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy', n_jobs=self.n_jobs)
            return scores.mean()
        
        # Optimize
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
        
        # Train best model
        best_params = study.best_params
        model_name = best_params.pop('model')
        
        if model_name == 'lightgbm':
            best_params['verbosity'] = -1
            best_params['n_jobs'] = self.n_jobs
            model = LGBMClassifier(**best_params)
        elif model_name == 'xgboost':
            best_params['tree_method'] = 'gpu_hist' if self.use_gpu else 'hist'
            best_params['verbosity'] = 0
            best_params['n_jobs'] = self.n_jobs
            model = XGBClassifier(**best_params)
        else:
            best_params['n_jobs'] = self.n_jobs
            model = RandomForestClassifier(**best_params)
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC for binary classification
        if len(np.unique(y_train)) == 2:
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            except:
                pass
        
        logger.info(f"Best model: {model_name} | Accuracy: {metrics['accuracy']:.4f}")
        
        return model, metrics
    
    def _train_regression(self, X_train, X_test, y_train, y_test, optuna_trials, cv_folds, progress_callback):
        """Train regression model"""
        
        # Define objective for Optuna
        def objective(trial):
            model_name = trial.suggest_categorical('model', ['lightgbm', 'xgboost', 'random_forest'])
            
            if model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'verbosity': -1,
                    'n_jobs': self.n_jobs
                }
                model = LGBMRegressor(**params)
            
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
                    'verbosity': 0,
                    'n_jobs': self.n_jobs
                }
                model = XGBRegressor(**params)
            
            else:  # random_forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'n_jobs': self.n_jobs
                }
                model = RandomForestRegressor(**params)
            
            # Cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=self.n_jobs)
            return -scores.mean()  # Negative because we want to minimize MSE
        
        # Optimize
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
        
        # Train best model
        best_params = study.best_params
        model_name = best_params.pop('model')
        
        if model_name == 'lightgbm':
            best_params['verbosity'] = -1
            best_params['n_jobs'] = self.n_jobs
            model = LGBMRegressor(**best_params)
        elif model_name == 'xgboost':
            best_params['tree_method'] = 'gpu_hist' if self.use_gpu else 'hist'
            best_params['verbosity'] = 0
            best_params['n_jobs'] = self.n_jobs
            model = XGBRegressor(**best_params)
        else:
            best_params['n_jobs'] = self.n_jobs
            model = RandomForestRegressor(**best_params)
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Best model: {model_name} | RMSE: {metrics['rmse']:.4f} | R2: {metrics['r2_score']:.4f}")
        
        return model, metrics