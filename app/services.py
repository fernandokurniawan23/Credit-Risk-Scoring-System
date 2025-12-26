"""
app/services.py
Service Layer: Handles communication with the Backend API.
Decouples business logic from UI logic.
"""
import requests
from typing import Dict, Any, Optional

class CreditRiskService:
    def __init__(self, api_url: str = "http://127.0.0.1:8000/predict"):
        self.api_url = api_url

    def get_prediction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends application data to the inference API.
        
        Args:
            payload: Dictionary containing applicant data.
            
        Returns:
            Dict containing prediction results (score, tier, etc.)
            
        Raises:
            Exception: If API connection fails or returns non-200.
        """
        try:
            response = requests.post(self.api_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to Scoring Engine. Ensure API is running.")