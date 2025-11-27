from pydantic import BaseModel

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
