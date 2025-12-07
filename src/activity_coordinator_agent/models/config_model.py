from pydantic import BaseModel, Field

class Config(BaseModel):
    model: str = Field(..., description="The name of the language model to use.")
    temperature: float = Field(0.7, description="The temperature setting for the language model.")