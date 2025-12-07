from ..models.context_model import Context
from ollama import chat, ChatResponse
import json

SYSTEM_PROMPT = f"""
You are a helpful assistant that extracts care-home activity context from natural language.
You will receive user messages and your task is to identify and extract relevant context information related to care-home activities.
Extract the context in the JSON format that conforms to the schema specified below:
```
{json.dumps(Context.model_json_schema(), indent=2)}
```

You must strictly adhere to the following guidelines when extracting context and only include json only
"""


class ContextAgent:
    def __init__(self, system_prompt: str | None = None):
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        self.system_prompt = system_prompt

    def run(self, user_prompt: str, model_id: str = 'qwen3:8b') -> Context:
        response: ChatResponse = chat(
            model=model_id,
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            options={"temperature": 0.0}
        )

        try:
            response_text = json.loads(response.message.content)
            context = Context(**response_text)
        except Exception as e:
            from .json_correction_agent import JsonCorrectionAgent

            json_correction_agent = JsonCorrectionAgent()
            corrected_json = json_correction_agent.run(
                text_response=response.message.content,
                json_schema=Context,
                error_message=str(e),
            )

            context = Context(**corrected_json)

        return context