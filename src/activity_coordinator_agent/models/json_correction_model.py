from pydantic import BaseModel, Field
from typing import Optional


class JsonCorrection(BaseModel):
    text_response: str = Field(..., description="The text response from the LLM containing invalid JSON data.")
    json_schema: str = Field(..., description="The JSON schema that the corrected JSON data should conform to.")
    error_message: Optional[str] = Field("not provided", description="The error message indicating why the JSON is invalid.")