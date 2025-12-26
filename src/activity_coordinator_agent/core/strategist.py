from google import genai
from google.genai import types
from ..schemas.strategist import ActivityStrategy
import json

class StrategistLogic:
    def __init__(
            self,
            client: genai.Client,
            model: str = "gemini-2.5-flash"
    ):
        self.client = client
        self.model = model
    
    @property
    def output_schema(self):
        return ActivityStrategy

    @property
    def system_prompt(self) -> str:return """
        ### ROLE
        You are an expert Social Engagement Strategist for dementia care.

        ### TASK
        You will receive a 'Patient Profile' and an 'Activity Profile'. 
        Identify "Semantic Bridges" (Hooks) to help the carer engage the patient.

        ### BRIDGE TYPES (Prioritize in this order):
        1. **Direct History:** The activity matches a past hobby/career (e.g., Knitting Activity <-> Seamstress).
        2. **Skill/Career Transfer:** Abstracting a skill (e.g., Painting <-> Architect's eye for lines).
        3. **Sensory/Emotional:** Matching preferences (e.g., Music Session <-> "Loves relaxing atmospheres").
        4. **Constraint Adaptation (CRITICAL):** If a patient dislikes an aspect of the activity (e.g., "Gardening" but "Hates Dirt"), you MUST create a hook that accommodates this (e.g., "Focus on visual appreciation rather than digging").

        ### OUTPUT
        Return a structured list of Hooks with specific strategies.
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

from ..schemas.resident_profile import ResidentProfile
from ..schemas.activity_profile import ActivityProfile

def generate_strategy(client: genai.Client, resident: ResidentProfile, activity: ActivityProfile) -> ActivityStrategy:

    agent = StrategistLogic(client=client)
    
    # We combine the two structured inputs into a clear text representation for the prompt
    combined_input = f"""
    === RESIDENT PROFILE ===
    {resident.model_dump_json()}
    
    === ACTIVITY PROFILE ===
    {activity.model_dump_json()}
    """
    
    return agent.run(combined_input)

if __name__ == "__main__":
    # Test without running the whole app
    import dotenv
    dotenv.load_dotenv()

    client = genai.Client()

