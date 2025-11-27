"""
Model Training Script
Train and evaluate cryptocurrency prediction models
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime
import time
import json
import argparse
from typing import Dict, Any, List, Tuple

from app.services.preprocessing import DataPreprocessor, create_feature_engineering_pipeline
from app.models.ml_model import CryptoPredictor, ModelEnsemble


def load_and_prepare_data(
    data_path: str,
    add_features: bool = True
) -> Tuple[pd.DataFrame, DataPreprocessor]:
    """
    Load and prepare data for training.
    
    Args:
        data_path: Path to CSV data file
        add_features: Whether to add engineered features
        
    Returns:
        Tuple of (prepared dataframe, preprocessor)
    """
    print("=" * 60)
    print("LOADING AND PREPARING DATA")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(scaler_type='standard')
    
    # Load data
    df = preprocessor.load_data(data_path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Clean data
    df = preprocessor.clean_data(df)
    print(f"After cleaning: {len(df)} rows")
    
    # Optional feature engineering
    if add_features:
        df = create_feature_engineering_pipeline(df)
        print(f"After feature engineering: {len(df)} rows, {len(df.columns)} columns")
    
    # Check class balance
    balance = preprocessor.check_class_balance(df['Target'].values)
    print(f"\nClass distribution:")
    for cls, count, pct in zip(balance['classes'], balance['counts'], balance['percentages']):
        print(f"  Class {cls}: {count} samples ({pct:.1f}%)")
    print(f"  Balanced: {balance['is_balanced']}")
    
    return df, preprocessor


def train_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str]
) -> CryptoPredictor:
    """
    Train a single model.
    
    Args:
        model_type: Type of model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_names: List of feature names
        
    Returns:
        Trained model
    """
    print(f"\nTraining {model_type}...")
    start_time = time.time()
    
    model = CryptoPredictor(model_type=model_type)
    model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.2f} seconds")
    
    return model


def evaluate_model(
    model: CryptoPredictor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate model on all datasets.
    
    Returns:
        Dictionary of metrics for each dataset
    """
    metrics = {}
    
    for name, X, y in [
        ('train', X_train, y_train),
        ('validation', X_val, y_val),
        ('test', X_test, y_test)
    ]:
        m = model.evaluate(X, y, dataset_name=name)
        metrics[name] = m
        print(f"\n  {name.upper()} Metrics:")
        print(f"    Accuracy:  {m['accuracy']:.4f}")
        print(f"    Precision: {m['precision']:.4f}")
        print(f"    Recall:    {m['recall']:.4f}")
        print(f"    F1-Score:  {m['f1_score']:.4f}")
        print(f"    ROC-AUC:   {m['roc_auc']:.4f}")
    
    return metrics


def train_all_models(
    data_path: str,
    output_dir: str,
    model_types: List[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict[str, Any]:
    """
    Train and compare multiple models.
    
    Args:
        data_path: Path to data file
        output_dir: Directory to save models
        model_types: List of model types to train
        train_ratio: Training data proportion
        val_ratio: Validation data proportion
        
    Returns:
        Training results summary
    """
    if model_types is None:
        model_types = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    df, preprocessor = load_and_prepare_data(data_path)
    
    # Prepare features
    feature_columns = preprocessor.get_feature_columns(df)
    X, y = preprocessor.prepare_features(df, feature_columns)
    
    print(f"\nFeatures: {len(feature_columns)}")
    print(f"Samples: {len(X)}")
    
    # Split data (time-series aware)
    splits = preprocessor.time_series_split(X, y, train_ratio, val_ratio)
    
    print(f"\nData splits:")
    print(f"  Train:      {len(splits['X_train'])} samples")
    print(f"  Validation: {len(splits['X_val'])} samples")
    print(f"  Test:       {len(splits['X_test'])} samples")
    
    # Scale features
    X_train_scaled = preprocessor.fit_scaler(splits['X_train'])
    X_val_scaled = preprocessor.transform(splits['X_val'])
    X_test_scaled = preprocessor.transform(splits['X_test'])
    
    # Save preprocessor
    preprocessor_path = output_path / 'preprocessor.joblib'
    preprocessor.save(str(preprocessor_path))
    
    # Train and evaluate models
    results = {}
    best_model = None
    best_score = 0
    
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    for model_type in model_types:
        print(f"\n{'=' * 40}")
        print(f"Model: {model_type.upper()}")
        print('=' * 40)
        
        try:
            # Train
            model = train_model(
                model_type,
                X_train_scaled, splits['y_train'],
                X_val_scaled, splits['y_val'],
                feature_columns
            )
            
            # Evaluate
            metrics = evaluate_model(
                model,
                X_train_scaled, splits['y_train'],
                X_val_scaled, splits['y_val'],
                X_test_scaled, splits['y_test']
            )
            
            # Track best model (by validation F1)
            val_f1 = metrics['validation']['f1_score']
            if val_f1 > best_score:
                best_score = val_f1
                best_model = model
            
            # Save model
            model_path = output_path / f'{model_type}_model.joblib'
            model.save(str(model_path))
            
            results[model_type] = {
                'metrics': metrics,
                'model_path': str(model_path),
                'success': True
            }
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[model_type] = {
                'success': False,
                'error': str(e)
            }
    
    # Save best model as default
    if best_model:
        best_model_path = output_path / 'crypto_model.joblib'
        best_model.save(str(best_model_path))
        print(f"\n{'=' * 60}")
        print(f"BEST MODEL: {best_model.model_type}")
        print(f"Validation F1: {best_score:.4f}")
        print(f"Saved to: {best_model_path}")
        print('=' * 60)
        
        # Print feature importance
        importance = best_model.get_feature_importance()
        if importance:
            print("\nTop 10 Feature Importances:")
            for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
                print(f"  {i}. {feature}: {score:.4f}")
    
    # Save results summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'data_path': data_path,
        'n_samples': len(X),
        'n_features': len(feature_columns),
        'feature_names': feature_columns,
        'train_samples': len(splits['X_train']),
        'val_samples': len(splits['X_val']),
        'test_samples': len(splits['X_test']),
        'best_model': best_model.model_type if best_model else None,
        'best_validation_f1': best_score,
        'results': results
    }
    
    summary_path = output_path / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nTraining summary saved to: {summary_path}")
    
    return summary


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train cryptocurrency prediction models')
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/dataset.csv',
        help='Path to dataset CSV'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='ml/saved_models',
        help='Output directory for models'
    )
    parser.add_argument(
        '--models', 
        type=str, 
        nargs='+',
        default=['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'],
        help='Model types to train'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training data ratio'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation data ratio'
    )
    
    args = parser.parse_args()
    
    # Run training
    results = train_all_models(
        data_path=args.data,
        output_dir=args.output,
        model_types=args.models,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    return results


if __name__ == '__main__':
    main()