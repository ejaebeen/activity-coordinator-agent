from typing import List, Literal
from pydantic import BaseModel, Field

class ConnectionHook(BaseModel):
    """Represents a single bridge between patient and activity."""
    topic: str = Field(..., description="The shared subject (e.g., 'Vegetables', 'Soil').")
    connection_type: Literal["Direct History", "Sensory/Emotional", "Skill/Career", "Constraint Adaptation"]
    strategy: str = Field(..., description="The angle to take. (e.g., 'Ask about harvest time', 'Focus on visual observation to avoid dirt').")

class ActivityStrategy(BaseModel):
    """The full output of the Strategist Node."""
    reasoning: str = Field(..., description="Brief analysis of why these hooks were chosen.")
    hooks: List[ConnectionHook] = Field(..., description="List of 3-5 distinct conversation bridges.")