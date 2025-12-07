from pydantic import BaseModel, Field
from typing import Optional


class Context(BaseModel):
    resident_profile: Optional[str] = Field(None, description="general description of physical and cognitive abilities of the residents in the activity")
    n_residents: Optional[str] = Field(None, description="approximate number of residents in the care home")
    duration: Optional[str] = Field(None, description="duration of the activity session.")
    activity_type: Optional[str] = Field(None, description="type of activity e.g., music, movement, art, games, social etc.")
