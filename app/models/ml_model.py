"""
Machine Learning Model Wrapper
Provides unified interface for different ML models
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Any, Optional, List, Tuple, Union
import joblib
from pathlib import Path
from datetime import datetime
import json


class CryptoPredictor:
    """
    Unified wrapper for cryptocurrency price direction prediction models.
    Supports multiple algorithms with consistent interface.
    """
    
    SUPPORTED_MODELS = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'xgboost': XGBClassifier,
        'lightgbm': LGBMClassifier
    }
    
    DEFAULT_PARAMS = {
        'logistic_regression': {
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced'
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1
        }
    }
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model to use
            params: Custom hyperparameters (optional)
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model type '{model_type}' not supported. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        self.model_type = model_type
        self.params = params or self.DEFAULT_PARAMS.get(model_type, {})
        self.model = self.SUPPORTED_MODELS[model_type](**self.params)
        self.feature_names = None
        self.metrics = {}
        self.trained = False
        self.training_date = None
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'CryptoPredictor':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: List of feature names (optional)
            
        Returns:
            Self for method chaining
        """
        self.feature_names = feature_names
        
        # Some models support early stopping with validation set
        if X_val is not None and y_val is not None:
            if self.model_type == 'xgboost':
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif self.model_type == 'lightgbm':
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)]
                )
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        self.trained = True
        self.training_date = datetime.now().isoformat()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = 'test'
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            dataset_name: Name for this evaluation (e.g., 'train', 'val', 'test')
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, y_proba)),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
        
        self.metrics[dataset_name] = metrics
        return metrics
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.trained:
            return None
        
        # Get importance based on model type
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return None
        
        # Create feature name mapping
        if self.feature_names is not None:
            feature_importance = dict(zip(self.feature_names, importances))
        else:
            feature_importance = {f'feature_{i}': imp for i, imp in enumerate(importances)}
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return feature_importance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and information."""
        return {
            'model_type': self.model_type,
            'parameters': self.params,
            'trained': self.trained,
            'training_date': self.training_date,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
    
    def save(self, filepath: str):
        """
        Save model and metadata to disk.
        
        Args:
            filepath: Path to save model
        """
        save_data = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'trained': self.trained,
            'training_date': self.training_date
        }
        
        joblib.dump(save_data, filepath)
        print(f"Model saved to {filepath}")
        
        # Also save metadata as JSON for easy inspection
        metadata_path = filepath.replace('.joblib', '_metadata.json')
        metadata = {
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'trained': self.trained,
            'training_date': self.training_date
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CryptoPredictor':
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded CryptoPredictor instance
        """
        save_data = joblib.load(filepath)
        
        predictor = cls(
            model_type=save_data['model_type'],
            params=save_data['params']
        )
        predictor.model = save_data['model']
        predictor.feature_names = save_data['feature_names']
        predictor.metrics = save_data['metrics']
        predictor.trained = save_data['trained']
        predictor.training_date = save_data['training_date']
        
        return predictor


class ModelEnsemble:
    """
    Ensemble of multiple models for improved predictions.
    Uses voting or averaging for final predictions.
    """
    
    def __init__(self, models: List[CryptoPredictor], voting: str = 'soft'):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained CryptoPredictor instances
            voting: 'hard' for majority voting, 'soft' for probability averaging
        """
        self.models = models
        self.voting = voting
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if self.voting == 'hard':
            predictions = np.array([m.predict(X) for m in self.models])
            return np.round(np.mean(predictions, axis=0)).astype(int)
        else:
            probas = np.array([m.predict_proba(X)[:, 1] for m in self.models])
            avg_proba = np.mean(probas, axis=0)
            return (avg_proba >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities."""
        probas = np.array([m.predict_proba(X) for m in self.models])
        return np.mean(probas, axis=0)