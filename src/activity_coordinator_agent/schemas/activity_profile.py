from typing import Literal
from pydantic import BaseModel, Field

class ActivityProfile(BaseModel):
    name: str = Field(description="Standardized name of the activity")
    
    # 1. Categorization (Helps matching interests)
    category: Literal["Creative", "Physical", "Music", "Social", "Reminiscence", "Sensory"]
    tags: list[str] = Field(description="Keywords e.g., ['Watercolors', 'Landscapes', 'Hand-eye coordination']")
    
    # 2. Capability Requirements (Crucial for matching resident level)
    cognitive_demand: Literal["High", "Moderate", "Low"] = Field(
        description="High: Rules/Memory required. Low: Passive observation."
    )
    physical_demand: Literal["Seated", "Standing", "Walking", "Active"]
    
    # 3. Sensory Profile (Crucial for safety/engagement)
    dominant_sense: Literal["Visual", "Auditory", "Tactile", "Olfactory", "Gustatory"]
    noise_level: Literal["Quiet", "Moderate", "Loud"]
    
    # 4. Content Analysis
    reminiscence_era: str | None = Field(
        default=None, 
        description="If relevant, the specific decade or era (e.g., '1950s', 'Wartime'). Useful for history matching."
    )