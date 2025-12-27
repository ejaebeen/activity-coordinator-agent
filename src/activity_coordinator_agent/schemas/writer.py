from pydantic import BaseModel, Field
from typing import Optional

class DraftOutput(BaseModel):
    drafts: list[str] = Field(description="List of 5 draft conversation starters")

class QuestionList(BaseModel):
    """The final output for the carer."""
    introduction: str = Field(description="A warm opening sentence to set the scene.")
    questions: list[str] = Field(description="5 conversation starters, specifically tailored to the hooks.")
    safety_note: Optional[str] = Field(description="A final reminder of any specific safety constraints used.")