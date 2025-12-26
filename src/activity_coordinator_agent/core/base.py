# src/care_connect/core/base.py
import os
import json
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Generic
import google.genai as genai
from pydantic import BaseModel

# A Generic Type Variable to say "This class returns some kind of Pydantic Model"
T = TypeVar('T', bound=BaseModel)

class BaseCoreLogic(ABC, Generic[T]):
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """
        Base class for all Core Logic nodes. 
        Handles Google Auth and JSON Schema enforcement.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model_name = model_name

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Define the persona and instructions here."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> Type[T]:
        """Define which Pydantic model this agent must return."""
        pass

    def run(self, user_input: str) -> T:
        """
        Executes the LLM call with strict JSON enforcement.
        """
        # 1. Initialize the Model
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt
        )

        # 2. Configure Structured Output (The Google Native Way)
        # We allow standard text output but force JSON MIME type
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=self.output_schema # Pass the class directly!
        )

        # 3. Generate Content
        response = model.generate_content(
            user_input,
            generation_config=generation_config
        )

        # 4. Parse & Validate
        # Google returns a JSON string. We parse it into the Pydantic model.
        try:
            # response.text is the raw JSON string
            return self.output_schema.model_validate_json(response.text)
        except Exception as e:
            print(f"Failed to parse JSON: {response.text}")
            raise e