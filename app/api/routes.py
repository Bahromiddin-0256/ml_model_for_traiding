"""
API Routes
FastAPI endpoint definitions
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import time

from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    ModelInfo,
    FeatureImportance,
    FeatureImportanceResponse,
    HealthCheck,
    ErrorResponse,
    TrainingRequest,
    TrainingResponse
)
from app.services.predictor import get_prediction_service, PredictionService


router = APIRouter()


def get_service() -> PredictionService:
    """Dependency to get prediction service."""
    return get_prediction_service()


@router.get("/health", response_model=HealthCheck, tags=["System"])
async def health_check(service: PredictionService = Depends(get_service)):
    """
    Check API health status.
    
    Returns service status and model availability.
    """
    health = service.health_check()
    return HealthCheck(**health)


@router.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info(service: PredictionService = Depends(get_service)):
    """
    Get information about the loaded model.
    
    Returns model type, training date, metrics, and feature information.
    """
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train or load a model first."
        )
    
    info = service.get_model_info()
    return ModelInfo(
        model_type=info['model_type'],
        trained=info['trained'],
        training_date=info.get('training_date'),
        n_features=info.get('n_features', 0),
        feature_names=info.get('feature_names', []),
        metrics=info.get('metrics', {})
    )


@router.get("/features", response_model=List[str], tags=["Model"])
async def get_required_features(service: PredictionService = Depends(get_service)):
    """
    Get list of required input features.
    
    Returns the feature names that must be provided for predictions.
    """
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return service.get_required_features()


@router.get(
    "/model/importance",
    response_model=FeatureImportanceResponse,
    tags=["Model"]
)
async def get_feature_importance(service: PredictionService = Depends(get_service)):
    """
    Get feature importance rankings.
    
    Returns features sorted by their importance to the model.
    """
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    importance = service.get_feature_importance()
    
    if importance is None:
        raise HTTPException(
            status_code=400,
            detail="Feature importance not available for this model type"
        )
    
    importances = [
        FeatureImportance(
            feature_name=name,
            importance=score,
            rank=rank
        )
        for rank, (name, score) in enumerate(importance.items(), 1)
    ]
    
    model_info = service.get_model_info()
    
    return FeatureImportanceResponse(
        importances=importances,
        model_type=model_info['model_type']
    )


@router.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(
    input_data: PredictionInput,
    service: PredictionService = Depends(get_service)
):
    """
    Make a single prediction.
    
    Provide technical indicators and receive a price direction prediction.
    
    - **prediction**: 1 = UP, 0 = DOWN
    - **probability_up/down**: Confidence scores for each direction
    - **confidence**: Overall model confidence (highest probability)
    """
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train or load a model first."
        )
    
    try:
        # Convert input to dictionary
        data = input_data.model_dump(exclude_none=True)
        
        # Make prediction
        result = service.predict(data)
        
        return PredictionOutput(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_batch(
    input_data: BatchPredictionInput,
    service: PredictionService = Depends(get_service)
):
    """
    Make batch predictions.
    
    Provide multiple sets of technical indicators for bulk predictions.
    Maximum 1000 predictions per request.
    """
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        results = service.predict_batch(input_data.data)
        
        # Filter successful predictions
        predictions = [
            PredictionOutput(**r) for r in results 
            if 'error' not in r
        ]
        
        # Calculate summary
        up_count = sum(1 for p in predictions if p.prediction == 1)
        down_count = len(predictions) - up_count
        
        return BatchPredictionOutput(
            predictions=predictions,
            total_count=len(predictions),
            summary={
                'up': up_count,
                'down': down_count,
                'errors': len(results) - len(predictions)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/model/train", response_model=TrainingResponse, tags=["Model"])
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    service: PredictionService = Depends(get_service)
):
    """
    Train a new model.
    
    This endpoint triggers model training with the specified parameters.
    Training runs in the background for large datasets.
    """
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from ml.train import train_all_models
    
    start_time = time.time()
    
    try:
        # Run training
        data_path = str(project_root / 'data' / 'dataset.csv')
        output_dir = str(project_root / 'ml' / 'saved_models')
        
        results = train_all_models(
            data_path=data_path,
            output_dir=output_dir,
            model_types=[request.model_type],
            train_ratio=1 - request.test_size - request.validation_size,
            val_ratio=request.validation_size
        )
        
        training_time = time.time() - start_time
        
        # Reload model in service
        model_path = Path(output_dir) / 'crypto_model.joblib'
        preprocessor_path = Path(output_dir) / 'preprocessor.joblib'
        
        service.load_model(str(model_path))
        service.load_preprocessor(str(preprocessor_path))
        
        # Get metrics for response
        model_results = results['results'].get(request.model_type, {})
        metrics = model_results.get('metrics', {})
        
        return TrainingResponse(
            success=True,
            model_type=request.model_type,
            training_samples=results['train_samples'],
            validation_samples=results['val_samples'],
            test_samples=results['test_samples'],
            metrics=metrics,
            training_time_seconds=training_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@router.get("/model/metrics", tags=["Model"])
async def get_model_metrics(service: PredictionService = Depends(get_service)):
    """
    Get detailed model performance metrics.
    
    Returns accuracy, precision, recall, F1, and ROC-AUC for all datasets.
    """
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    info = service.get_model_info()
    return {
        'model_type': info['model_type'],
        'metrics': info.get('metrics', {})
    }