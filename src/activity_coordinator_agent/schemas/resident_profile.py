from pydantic import BaseModel, Field

# This is the Best Practice: Define the output as a Class
class ResidentProfile(BaseModel):
    reasoning_scratchpad: str = Field(..., description="Internal thought process analyzing safety and cognition.")
    cognitive_level: str = Field(
        ..., 
        description="The resident's cognitive capability. Options: 'High' (Conversational), 'Moderate' (Some confusion), 'Low' (Sensory/Non-verbal)."
    )
    hobbies: list[str] = Field(
        description="List of 3-5 distinct interests, hobbies, or past careers. Extract specific nouns (e.g., 'Knitting' not 'Doing things')."
    )
    sensitive_topics: list[str] = Field(
        default=[], 
        description="List of topics that cause distress or are flagged as 'do not mention'. If none found, return empty list."
    )
    communication_style: str = Field(
        ..., 
        description="Brief advice on how to speak to them (e.g., 'Use short sentences', 'Speak loudly')."
    )