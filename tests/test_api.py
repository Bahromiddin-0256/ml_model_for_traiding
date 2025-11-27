"""
API Tests
Test FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self):
        """Test health endpoint returns valid response."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data


class TestModelEndpoints:
    """Tests for model information endpoints."""
    
    def test_get_features_without_model(self):
        """Test features endpoint when no model is loaded."""
        response = client.get("/api/v1/features")
        # Should return 503 if no model loaded
        assert response.status_code in [200, 503]
    
    def test_get_model_info_without_model(self):
        """Test model info endpoint when no model is loaded."""
        response = client.get("/api/v1/model/info")
        # Should return 503 if no model loaded
        assert response.status_code in [200, 503]


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""
    
    def test_predict_without_model(self):
        """Test prediction endpoint when no model is loaded."""
        test_data = {
            "Open": 45000.0,
            "High": 46000.0,
            "Low": 44000.0,
            "Close": 45500.0,
            "Volume": 1000000.0
        }
        
        response = client.post("/api/v1/predict", json=test_data)
        # Should return 503 if no model loaded
        assert response.status_code in [200, 503]
    
    def test_predict_missing_fields(self):
        """Test prediction with missing required fields."""
        test_data = {
            "Open": 45000.0
            # Missing other required fields
        }
        
        response = client.post("/api/v1/predict", json=test_data)
        # Should return 422 for validation error or 503 if no model
        assert response.status_code in [422, 503]
    
    def test_batch_predict_without_model(self):
        """Test batch prediction endpoint."""
        test_data = {
            "data": [
                {
                    "Open": 45000.0,
                    "High": 46000.0,
                    "Low": 44000.0,
                    "Close": 45500.0,
                    "Volume": 1000000.0
                }
            ]
        }
        
        response = client.post("/api/v1/predict/batch", json=test_data)
        assert response.status_code in [200, 503]


class TestWebInterface:
    """Tests for web interface endpoints."""
    
    def test_home_page(self):
        """Test home page loads."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_dashboard_page(self):
        """Test dashboard page loads."""
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""
    
    def test_openapi_schema(self):
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_swagger_docs(self):
        """Test Swagger UI is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc(self):
        """Test ReDoc is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])