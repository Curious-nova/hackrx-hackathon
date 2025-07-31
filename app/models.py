# app/models.py

from pydantic import BaseModel, Field
from typing import List

class HackathonRequest(BaseModel):
    """Defines the structure of the incoming API request."""
    documents: str = Field(..., description="URL to the policy document.")
    questions: List[str] = Field(..., description="List of questions to answer.")

class HackathonResponse(BaseModel):
    """Defines the structure of the API response."""
    answers: List[str]