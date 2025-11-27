"""
Crypto Price Direction Predictor API
FastAPI Application Entry Point
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pathlib import Path
import logging

from app.api.routes import router as api_router
from app.services.predictor import get_prediction_service, initialize_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODELS_DIR = PROJECT_ROOT / "ml" / "saved_models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown."""
    # Startup
    logger.info("Starting Crypto Predictor API...")
    
    # Try to load existing model
    model_path = MODELS_DIR / "crypto_model.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    
    if model_path.exists():
        logger.info(f"Loading model from {model_path}")
        service = initialize_service(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path) if preprocessor_path.exists() else None
        )
        if service.is_ready():
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model loaded but not ready for predictions")
    else:
        logger.warning(f"No model found at {model_path}. Train a model first.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Crypto Predictor API...")


# Create FastAPI app
app = FastAPI(
    title="Crypto Price Direction Predictor",
    description="""
    Machine Learning API for predicting cryptocurrency price direction.
    
    ## Features
    
    * **Single Prediction**: Predict price direction from technical indicators
    * **Batch Prediction**: Process multiple predictions at once
    * **Model Training**: Train new models with different algorithms
    * **Model Insights**: View feature importance and performance metrics
    
    ## Models Supported
    
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM
    
    ## Technical Indicators
    
    The model uses various technical indicators including:
    - Moving Averages (SMA, EMA)
    - MACD
    - RSI
    - Bollinger Bands
    - Stochastic Oscillator
    - And many more...
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Setup templates
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def home(request: Request):
    """
    Render the main web interface.
    """
    service = get_prediction_service()
    
    context = {
        "request": request,
        "model_loaded": service.is_ready(),
        "model_info": service.get_model_info() if service.is_ready() else None
    }
    
    return templates.TemplateResponse("index.html", context)


@app.get("/dashboard", response_class=HTMLResponse, tags=["Web Interface"])
async def dashboard(request: Request):
    """
    Render the analytics dashboard.
    """
    service = get_prediction_service()
    
    context = {
        "request": request,
        "model_loaded": service.is_ready(),
        "model_info": service.get_model_info() if service.is_ready() else None,
        "feature_importance": service.get_feature_importance() if service.is_ready() else None
    }
    
    return templates.TemplateResponse("dashboard.html", context)


# Additional error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "error_code": 404, "message": "Page not found"},
        status_code=404
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "error_code": 500, "message": "Internal server error"},
        status_code=500
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )