"""
Credit Scoring API
------------------
Exposes the CreditScoringModel via REST endpoints.
Features:
1. Input validation using Pydantic.
2. Hot-loading of the model.
3. Health check endpoint.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
import os

# Add project root to sys.path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import CreditScoringModel

# Initialize App & Model
app = FastAPI(title="Credit Risk Scoring API", version="1.0")
model_service = CreditScoringModel()

# --- INPUT SCHEMA (Data Contract) ---
# Validates data types before it hits the model
class LoanApplication(BaseModel):
    SK_ID_CURR: int
    NAME_CONTRACT_TYPE: str
    CODE_GENDER: str
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    DAYS_EMPLOYED: int
    DAYS_BIRTH: int
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float

@app.get("/")
def health_check():
    """Simple health check to ensure API is running."""
    return {"status": "ok", "service": "Credit Risk Scorer"}

@app.post("/predict")
def predict_credit_risk(application: LoanApplication):
    """
    Main endpoint. Receives loan application data and returns credit decision.
    """
    try:
        # Convert Pydantic object to Dict
        data = application.model_dump()
        
        # Run Inference
        result = model_service.predict(data)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run server locally
    uvicorn.run(app, host="0.0.0.0", port=8000)