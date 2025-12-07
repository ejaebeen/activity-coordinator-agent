from pydantic import BaseModel, Field

class ActivityIdea(BaseModel):
    title: str = Field(..., description="A short, catchy title for the activity.")
    description: str = Field(..., description="A brief description of the activity.")
    setup_difficulty: int = Field(..., description="Difficulty level to set up the activity (1-5).")
    items_needed: list[str] = Field(..., description="A list of items needed to conduct the activity.")