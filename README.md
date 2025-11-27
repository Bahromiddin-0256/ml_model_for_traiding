# Crypto Price Direction Predictor

A machine learning-powered API and web interface for predicting cryptocurrency price direction using technical indicators.

## Features

- **Multiple ML Models**: Support for Logistic Regression, Random Forest, XGBoost, LightGBM
- **REST API**: FastAPI-based endpoints for predictions
- **Web Interface**: Modern dark-themed UI for interactive predictions
- **Dashboard**: Analytics and model performance visualization
- **Batch Predictions**: Process multiple predictions at once
- **Auto-scaling**: Scaled features for improved model performance

## Project Structure

```
crypto_predictor/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   └── routes.py        # API endpoint definitions
│   ├── models/
│   │   ├── ml_model.py      # ML model wrapper classes
│   │   └── schemas.py       # Pydantic validation schemas
│   ├── services/
│   │   ├── predictor.py     # Prediction service
│   │   └── preprocessing.py # Data preprocessing
│   ├── static/              # CSS and JavaScript
│   └── templates/           # HTML templates
├── ml/
│   ├── train.py             # Model training script
│   └── saved_models/        # Trained model files
├── data/
│   └── dataset.csv          # Training data
├── tests/
│   └── test_api.py          # API tests
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model

```bash
cd crypto_predictor
python ml/train.py --data data/dataset.csv --output ml/saved_models
```

### 3. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the Application

- **Web Interface**: http://localhost:8000
- **Dashboard**: http://localhost:8000/dashboard
- **API Documentation**: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/model/info` | Model information |
| GET | `/api/v1/features` | Required features list |
| GET | `/api/v1/model/importance` | Feature importance |
| POST | `/api/v1/predict` | Single prediction |
| POST | `/api/v1/predict/batch` | Batch predictions |
| POST | `/api/v1/model/train` | Train new model |

## Prediction Request Example

```python
import requests

data = {
    "Open": 45000.0,
    "High": 46000.0,
    "Low": 44000.0,
    "Close": 45500.0,
    "Volume": 1500000000.0,
    "RSI": 55.5,
    "MACD": 150.5,
    "MACD_Signal": 120.3
}

response = requests.post("http://localhost:8000/api/v1/predict", json=data)
result = response.json()

print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Technical Indicators Used

- **Price Data**: Open, High, Low, Close, Volume
- **Moving Averages**: SMA (7, 25, 99), EMA (7, 25, 99)
- **MACD**: Line, Signal, Histogram
- **RSI**: Relative Strength Index
- **Bollinger Bands**: Upper, Middle, Lower
- **Stochastic Oscillator**: %K, %D
- **Others**: ATR, ADX, CCI, Williams %R, ROC, MFI, CMF, OBV, VWAP
- **Ichimoku Cloud**: Tenkan-sen, Kijun-sen, Senkou Spans, Chikou Span

## Docker Deployment

```bash
# Build image
docker build -t crypto-predictor .

# Run container
docker run -p 8000:8000 crypto-predictor
```

## Model Performance

The model is trained using time-series aware splitting to prevent data leakage:
- Training: 70%
- Validation: 15%
- Test: 15%

Typical performance metrics (varies by model):
- Accuracy: 52-58%
- F1 Score: 50-56%
- ROC-AUC: 52-60%

> Note: Cryptocurrency price prediction is inherently challenging. These models should be used for research/educational purposes only.

## License

MIT License