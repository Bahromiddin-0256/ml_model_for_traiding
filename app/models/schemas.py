"""
Pydantic Schemas for API Request/Response Validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
from datetime import datetime


class PredictionInput(BaseModel):
    """Input schema for single prediction request."""
    
    # Price data
    Open: float = Field(..., description="Opening price")
    High: float = Field(..., description="Highest price")
    Low: float = Field(..., description="Lowest price")
    Close: float = Field(..., description="Closing price")
    Volume: float = Field(..., ge=0, description="Trading volume")
    
    # Moving Averages
    SMA_7: Optional[float] = Field(None, description="7-day Simple Moving Average")
    SMA_25: Optional[float] = Field(None, description="25-day Simple Moving Average")
    SMA_99: Optional[float] = Field(None, description="99-day Simple Moving Average")
    EMA_7: Optional[float] = Field(None, description="7-day Exponential Moving Average")
    EMA_25: Optional[float] = Field(None, description="25-day Exponential Moving Average")
    EMA_99: Optional[float] = Field(None, description="99-day Exponential Moving Average")
    
    # MACD
    MACD: Optional[float] = Field(None, description="MACD line")
    MACD_Signal: Optional[float] = Field(None, description="MACD Signal line")
    MACD_Hist: Optional[float] = Field(None, description="MACD Histogram")
    
    # RSI
    RSI: Optional[float] = Field(None, ge=0, le=100, description="Relative Strength Index")
    
    # Bollinger Bands
    BB_Upper: Optional[float] = Field(None, description="Bollinger Band Upper")
    BB_Middle: Optional[float] = Field(None, description="Bollinger Band Middle")
    BB_Lower: Optional[float] = Field(None, description="Bollinger Band Lower")
    
    # Volatility
    ATR: Optional[float] = Field(None, ge=0, description="Average True Range")
    
    # Volume Indicators
    OBV: Optional[float] = Field(None, description="On-Balance Volume")
    
    # Momentum Indicators
    Stoch_K: Optional[float] = Field(None, description="Stochastic %K")
    Stoch_D: Optional[float] = Field(None, description="Stochastic %D")
    ADX: Optional[float] = Field(None, ge=0, description="Average Directional Index")
    CCI: Optional[float] = Field(None, description="Commodity Channel Index")
    Williams_R: Optional[float] = Field(None, description="Williams %R")
    ROC: Optional[float] = Field(None, description="Rate of Change")
    MFI: Optional[float] = Field(None, ge=0, le=100, description="Money Flow Index")
    CMF: Optional[float] = Field(None, description="Chaikin Money Flow")
    
    # Other Indicators
    VWAP: Optional[float] = Field(None, description="Volume Weighted Average Price")
    Parabolic_SAR: Optional[float] = Field(None, description="Parabolic SAR")
    
    # Ichimoku Cloud
    Ichimoku_Tenkan: Optional[float] = Field(None, description="Ichimoku Tenkan-sen")
    Ichimoku_Kijun: Optional[float] = Field(None, description="Ichimoku Kijun-sen")
    Ichimoku_SenkouA: Optional[float] = Field(None, description="Ichimoku Senkou Span A")
    Ichimoku_SenkouB: Optional[float] = Field(None, description="Ichimoku Senkou Span B")
    Ichimoku_Chikou: Optional[float] = Field(None, description="Ichimoku Chikou Span")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Open": 45000.0,
                "High": 46500.0,
                "Low": 44500.0,
                "Close": 46000.0,
                "Volume": 1500000000.0,
                "RSI": 55.5,
                "MACD": 150.5,
                "MACD_Signal": 120.3,
                "MACD_Hist": 30.2,
                "ATR": 1200.5,
                "ADX": 25.3
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction response."""
    
    prediction: int = Field(..., description="Predicted direction: 1=UP, 0=DOWN")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability_up: float = Field(..., ge=0, le=1, description="Probability of price going up")
    probability_down: float = Field(..., ge=0, le=1, description="Probability of price going down")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence (max probability)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "prediction_label": "UP",
                "probability_up": 0.73,
                "probability_down": 0.27,
                "confidence": 0.73,
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class BatchPredictionInput(BaseModel):
    """Input schema for batch prediction request."""
    
    data: List[Dict[str, float]] = Field(
        ..., 
        min_length=1,
        max_length=1000,
        description="List of feature dictionaries"
    )


class BatchPredictionOutput(BaseModel):
    """Output schema for batch prediction response."""
    
    predictions: List[PredictionOutput] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    summary: Dict[str, int] = Field(..., description="Summary of prediction counts")


class ModelInfo(BaseModel):
    """Model information and metadata."""
    
    model_type: str = Field(..., description="Type of ML model")
    version: str = Field(default="1.0.0", description="Model version")
    trained: bool = Field(..., description="Whether model is trained")
    training_date: Optional[str] = Field(None, description="Date model was trained")
    n_features: int = Field(..., description="Number of input features")
    feature_names: List[str] = Field(..., description="List of feature names")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Model performance metrics")


class FeatureImportance(BaseModel):
    """Feature importance scores."""
    
    feature_name: str = Field(..., description="Name of the feature")
    importance: float = Field(..., ge=0, description="Importance score")
    rank: int = Field(..., ge=1, description="Importance rank")


class FeatureImportanceResponse(BaseModel):
    """Response containing feature importance rankings."""
    
    importances: List[FeatureImportance] = Field(..., description="Sorted feature importances")
    model_type: str = Field(..., description="Model type used")


class HealthCheck(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


class TrainingRequest(BaseModel):
    """Request to train a new model."""
    
    model_type: str = Field(
        default="xgboost",
        description="Type of model to train"
    )
    test_size: float = Field(
        default=0.15,
        ge=0.1,
        le=0.4,
        description="Proportion of data for testing"
    )
    validation_size: float = Field(
        default=0.15,
        ge=0.1,
        le=0.3,
        description="Proportion of data for validation"
    )
    
    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        allowed = ['logistic_regression', 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
        if v not in allowed:
            raise ValueError(f"model_type must be one of {allowed}")
        return v


class TrainingResponse(BaseModel):
    """Response after model training."""
    
    success: bool = Field(..., description="Whether training succeeded")
    model_type: str = Field(..., description="Type of model trained")
    training_samples: int = Field(..., description="Number of training samples")
    validation_samples: int = Field(..., description="Number of validation samples")
    test_samples: int = Field(..., description="Number of test samples")
    metrics: Dict[str, Dict[str, Any]] = Field(..., description="Performance metrics")
    training_time_seconds: float = Field(..., description="Training duration")