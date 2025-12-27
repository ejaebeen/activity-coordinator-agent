# src/care_connect/core/profiler.py
from google import genai
from google.genai import types
from ..schemas.resident_profile import ResidentProfile
from ..schemas.activity_profile import ActivityProfile
import json

class ResidentProfilerLogic:
    def __init__(
            self,
            client: genai.Client,
            model: str = "gemini-2.5-flash"
    ):
        self.client = client
        self.model = model
    
    @property
    def output_schema(self):
        return ResidentProfile

    @property
    def system_prompt(self) -> str:
        return """
        ### ROLE
        You are an expert Clinical Data Specialist and Safety Officer.

        ### TASK
        Analyze the provided resident bio. Extract the resident profile.

        ### INPUT
        Description of the resident bio.

        ### OUTPUT
        Return ONLY valid JSON matching the schema provided.
        """
    
    @property
    def generate_config(self):
        return types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            response_mime_type='application/json',
            response_schema=self.output_schema,
        )
    
    def run(self, user_input: str):
        """
        Executes the LLM call with strict JSON enforcement.
        """
        response = self.client.models.generate_content(
            model=self.model,
            config=self.generate_config,
            contents=user_input,
        )

        try:
            json_response = json.loads(response.parts[0].text)
            return self.output_schema(**json_response)
        except Exception as e:
            raise RuntimeError(str(e))


class ActivityProfilerLogic:
    def __init__(
            self,
            client: genai.Client,
            model: str = "gemini-2.5-flash"
    ):
        self.client = client
        self.model = model
    
    @property
    def output_schema(self):
        return ActivityProfile

    @property
    def system_prompt(self) -> str:
        return """
        ### ROLE
        You are an expert Clinical Data Specialist and Safety Officer.

        ### TASK
        Analyze the provided activity spec. Extract the activity profile.

        ### INPUT
        Description of the resident bio.

        ### OUTPUT
        Return ONLY valid JSON matching the schema provided.
        """
    
    @property
    def generate_config(self):
        return types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            response_mime_type='application/json',
            response_schema=self.output_schema,
        )
    
    def run(self, user_input: str):
        """
        Executes the LLM call with strict JSON enforcement.
        """
        response = self.client.models.generate_content(
            model=self.model,
            config=self.generate_config,
            contents=user_input,
        )

        try:
            json_response = json.loads(response.parts[0].text)
            return self.output_schema(**json_response)
        except Exception as e:
            raise RuntimeError(str(e))

        

# Convenience function for the rest of the app to call
def analyze_bio(bio_text: str, client: genai.Client = None):
    agent = ResidentProfilerLogic(client=client)
    return agent.run(bio_text)

# --- Direct Testing Block ---
if __name__ == "__main__":
    # Test without running the whole app
    dummy_bio = "Arthur (80). Loves jazz. gets agitated when people mention the war."
    import dotenv
    dotenv.load_dotenv()

    client = genai.Client()
    try:
        result = analyze_bio(dummy_bio, client)
        print("✅ SUCCESS")
        print(result)
    except Exception as e:
        print(f"❌ ERROR: {e}")