from google import genai
from google.genai import types
import json
from ..schemas.writer import QuestionList, DraftOutput
from ..schemas.strategist import ActivityStrategy
from ..schemas.resident_profile import ResidentProfile

class WriterLogic:
    def __init__(
            self,
            client: genai.Client,
            model: str = "gemini-2.5-flash"
    ):
        self.client = client
        self.model = model
    
    @property
    def output_schema(self):
        return DraftOutput

    @property
    def system_prompt(self) -> str:
        return """
        ### ROLE
        You are a Creative Conversation Designer for dementia care.
        
        ### TASK
        Using the provided 'Strategy Hooks', draft 5 distinct conversation starters.
        
        ### DRAFTING GUIDELINES
        1. **Use Scaffolding:** Never ask a raw question. State the context first.
           - *Bad:* "What was your favorite job?"
           - *Good (Hook: Career):* "I know you worked as a structural engineer for 40 years. That must have been fascinating. What was your favorite project?"
        
        2. **Vary the Logic:**
           - Question 1 & 2: Focus on **Long-term Memory** (The Past).
           - Question 3 & 4: Focus on **Sensory/Opinion** (The Present).
           - Question 5: Focus on **Imagination** (The "What if").

        ### OUTPUT
        Return exactly 5 draft strings.
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


class EditorLogic(WriterLogic):
    @property
    def output_schema(self):
        return QuestionList  # This is the final list + introduction

    @property
    def system_prompt(self) -> str:
        return f"""
        ### ROLE
        You are a Safety Compliance Officer and Editor.
        
        ### TASK
        Review the 5 draft questions provided. For each question, apply this 2-pass check:
        
        **Pass 1: Safety Check**
        - Does it touch on a Safety Flag? (e.g., asking about a deceased spouse when flagged as 'Grief Trigger').
        - **Action:** If UNSAFE, rewrite the question entirely to focus on a neutral topic (e.g., the weather, the current activity).
        
        **Pass 2: Tone Check**
        - Is it patronizing or "Elderspeak" (e.g., "Good boy", "Sweetie", "Let's take our meds")?
        - **Action:** If PATRONIZING, rewrite it to be adult-to-adult (e.g., "Arthur, could you help me with this?").
        
        ### OUTPUT
        - A warm 'introduction' sentence.
        - The final list of 5 validated/corrected questions.
        - A 'safety_note' explaining any edits you made (optional).
        """



def generate_questions(client: genai.Client, resident_profile: ResidentProfile, strategy: ActivityStrategy) -> QuestionList:
    agent = WriterLogic(client=client)
    
    # Format the input for the LLM
    prompt_input = f"""
    COGNITIVE LEVEL: {resident_profile.cognitive_level}
    
    CHOSEN STRATEGY HOOKS:
    {strategy.model_dump_json(indent=2)}
    """
    
    return agent.run(prompt_input)

def edit_questions(client: genai.Client, drafts: DraftOutput) -> QuestionList:
    agent = EditorLogic(client=client)
    return agent.run(drafts.model_dump_json(indent=2))