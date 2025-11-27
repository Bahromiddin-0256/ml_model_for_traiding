"""
Prediction Service
Handles model loading and inference
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from app.models.ml_model import CryptoPredictor
from app.services.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service for managing model loading and predictions.
    Implements singleton pattern for efficient model reuse.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None
    ):
        if self._initialized:
            return
            
        self.model: Optional[CryptoPredictor] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self._initialized = True
        
        # Auto-load if paths provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        if preprocessor_path and Path(preprocessor_path).exists():
            self.load_preprocessor(preprocessor_path)
    
    def load_model(self, path: str) -> bool:
        """Load trained model from disk."""
        try:
            self.model = CryptoPredictor.load(path)
            self.model_path = path
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def load_preprocessor(self, path: str) -> bool:
        """Load preprocessor from disk."""
        try:
            self.preprocessor = DataPreprocessor.load(path)
            self.preprocessor_path = path
            logger.info(f"Preprocessor loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if service is ready for predictions."""
        return self.model is not None and self.model.trained
    
    def get_required_features(self) -> List[str]:
        """Get list of required feature names."""
        if self.model and self.model.feature_names:
            return self.model.feature_names
        return []
    
    def _prepare_features(self, data: Dict[str, float]) -> np.ndarray:
        """
        Prepare input features for prediction.
        
        Args:
            data: Dictionary of feature values
            
        Returns:
            Numpy array ready for prediction
        """
        required_features = self.get_required_features()
        
        # Create feature array in correct order
        features = []
        for feature in required_features:
            value = data.get(feature)
            if value is None:
                raise ValueError(f"Missing required feature: {feature}")
            features.append(value)
        
        X = np.array(features).reshape(1, -1)
        
        # Apply scaling if preprocessor available
        if self.preprocessor and self.preprocessor.fitted:
            X = self.preprocessor.transform(X)
        
        return X
    
    def predict(self, data: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            data: Dictionary of feature values
            
        Returns:
            Prediction result dictionary
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded or not trained")
        
        X = self._prepare_features(data)
        
        # Get prediction and probabilities
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        return {
            'prediction': int(prediction),
            'prediction_label': 'UP' if prediction == 1 else 'DOWN',
            'probability_up': float(probabilities[1]),
            'probability_down': float(probabilities[0]),
            'confidence': float(max(probabilities)),
            'timestamp': datetime.now()
        }
    
    def predict_batch(self, data_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            data_list: List of feature dictionaries
            
        Returns:
            List of prediction results
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded or not trained")
        
        results = []
        for data in data_list:
            try:
                result = self.predict(data)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'prediction': None,
                    'timestamp': datetime.now()
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {
                'model_loaded': False,
                'message': 'No model loaded'
            }
        
        info = self.model.get_model_info()
        info['model_loaded'] = True
        info['model_path'] = self.model_path
        
        return info
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the model."""
        if not self.model:
            return None
        return self.model.get_feature_importance()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the service."""
        return {
            'status': 'healthy' if self.is_ready() else 'degraded',
            'model_loaded': self.model is not None,
            'model_trained': self.model.trained if self.model else False,
            'preprocessor_loaded': self.preprocessor is not None,
            'timestamp': datetime.now()
        }


# Global service instance
_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Get or create the prediction service singleton."""
    global _prediction_service
    
    if _prediction_service is None:
        # Default paths
        base_path = Path(__file__).parent.parent.parent / 'ml' / 'saved_models'
        model_path = base_path / 'crypto_model.joblib'
        preprocessor_path = base_path / 'preprocessor.joblib'
        
        _prediction_service = PredictionService(
            model_path=str(model_path) if model_path.exists() else None,
            preprocessor_path=str(preprocessor_path) if preprocessor_path.exists() else None
        )
    
    return _prediction_service


def initialize_service(model_path: str, preprocessor_path: Optional[str] = None):
    """Initialize the prediction service with specific paths."""
    global _prediction_service
    
    _prediction_service = PredictionService(
        model_path=model_path,
        preprocessor_path=preprocessor_path
    )
    
    return _prediction_service