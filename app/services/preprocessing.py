"""
Data Preprocessing Service
Handles data cleaning, feature engineering, and transformations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Dict, Any
import joblib
from pathlib import Path


class DataPreprocessor:
    """Handles all data preprocessing operations for crypto prediction model."""
    
    # Features to exclude from training (non-predictive)
    EXCLUDE_FEATURES = ['Date', 'Target']
    
    # Features that typically have high predictive power for crypto
    CORE_FEATURES = [
        'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR', 'ADX',
        'Stoch_K', 'Stoch_D', 'Williams_R', 'ROC', 'MFI', 'OBV'
    ]
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_columns = None
        self.fitted = False
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        df = pd.read_csv(filepath)
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Drop rows with NaN values (common in first rows due to indicator calculations)
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        dropped_rows = initial_rows - len(df_clean)
        
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with missing values")
        
        # Reset index after dropping rows
        df_clean = df_clean.reset_index(drop=True)
        
        return df_clean
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding target and date)."""
        feature_cols = [col for col in df.columns if col not in self.EXCLUDE_FEATURES]
        return feature_cols
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix X and target vector y.
        
        Args:
            df: Cleaned dataframe
            feature_columns: List of feature column names (optional)
            
        Returns:
            Tuple of (X, y) arrays
        """
        if feature_columns is None:
            feature_columns = self.get_feature_columns(df)
        
        self.feature_columns = feature_columns
        
        X = df[feature_columns].values
        y = df['Target'].values
        
        return X, y
    
    def fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """
        Fit scaler on training data and transform.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        
        return X_scaled
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit_scaler first.")
        
        return self.scaler.transform(X)
    
    def time_series_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Dict[str, np.ndarray]:
        """
        Split data maintaining temporal order (no shuffling for time series).
        
        Args:
            X: Feature matrix
            y: Target vector
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            Dictionary with train/val/test splits
        """
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        return {
            'X_train': X[:train_end],
            'X_val': X[train_end:val_end],
            'X_test': X[val_end:],
            'y_train': y[:train_end],
            'y_val': y[train_end:val_end],
            'y_test': y[val_end:]
        }
    
    def get_feature_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistical summary of features."""
        feature_cols = self.get_feature_columns(df)
        return df[feature_cols].describe()
    
    def check_class_balance(self, y: np.ndarray) -> Dict[str, Any]:
        """Check class distribution in target variable."""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        return {
            'classes': unique.tolist(),
            'counts': counts.tolist(),
            'percentages': (counts / total * 100).tolist(),
            'is_balanced': min(counts) / max(counts) > 0.8
        }
    
    def save(self, filepath: str):
        """Save preprocessor state (scaler and feature columns)."""
        state = {
            'scaler': self.scaler,
            'scaler_type': self.scaler_type,
            'feature_columns': self.feature_columns,
            'fitted': self.fitted
        }
        joblib.dump(state, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """Load preprocessor from saved state."""
        state = joblib.load(filepath)
        
        preprocessor = cls(scaler_type=state['scaler_type'])
        preprocessor.scaler = state['scaler']
        preprocessor.feature_columns = state['feature_columns']
        preprocessor.fitted = state['fitted']
        
        return preprocessor


def create_feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Additional feature engineering beyond base indicators.
    
    Args:
        df: Dataframe with base features
        
    Returns:
        Dataframe with additional engineered features
    """
    df_eng = df.copy()
    
    # Price momentum features
    if 'Close' in df.columns:
        df_eng['Price_Change'] = df_eng['Close'].pct_change()
        df_eng['Price_Change_Abs'] = df_eng['Price_Change'].abs()
        
    # Volatility features
    if 'ATR' in df.columns and 'Close' in df.columns:
        df_eng['ATR_Ratio'] = df_eng['ATR'] / df_eng['Close']
        
    # Bollinger Band position
    if all(col in df.columns for col in ['Close', 'BB_Upper', 'BB_Lower']):
        bb_range = df_eng['BB_Upper'] - df_eng['BB_Lower']
        df_eng['BB_Position'] = (df_eng['Close'] - df_eng['BB_Lower']) / bb_range
        
    # RSI extremes
    if 'RSI' in df.columns:
        df_eng['RSI_Overbought'] = (df_eng['RSI'] > 70).astype(int)
        df_eng['RSI_Oversold'] = (df_eng['RSI'] < 30).astype(int)
        
    # MACD signal
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        df_eng['MACD_Above_Signal'] = (df_eng['MACD'] > df_eng['MACD_Signal']).astype(int)
        
    # Volume features
    if 'Volume' in df.columns:
        df_eng['Volume_SMA_Ratio'] = df_eng['Volume'] / df_eng['Volume'].rolling(20).mean()
        
    # Drop any new NaN rows created by rolling calculations
    df_eng = df_eng.dropna()
    
    return df_eng