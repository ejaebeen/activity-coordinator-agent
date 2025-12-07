from ollama import chat, ChatResponse
from pydantic import BaseModel
import json

SYSTEM_PROMPT = """
You are a helpful assistant that helps correct invalid JSON data created from the responses of a LLM.
You will receive a text response of llm containing invalid JSON data and an error message indicating why the JSON is invalid.
Your task is to correct the JSON data so that it conforms to the specified JSON schema.
You must strictly adhere to the following guidelines when extracting context and only include json only

user prompt will be in the following format:

TEXT_RESPONSE: <text_response>

JSON_SCHEMA: <json_schema>

ERROR_MESSAGE: <error_message>
"""

USER_PROMPT_TEMPLATE = """
TEXT_RESPONSE: {text_response}

JSON_SCHEMA: {json_schema}

ERROR_MESSAGE: {error_message}
"""

class JsonCorrectionAgent:
    def __init__(
            self, 
            system_prompt: str | None = None,
            user_prompt_template: str | None = None,
        ):
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        self.system_prompt = system_prompt
        if user_prompt_template is None:
            user_prompt_template = USER_PROMPT_TEMPLATE
        self.user_prompt_template = user_prompt_template

    def _generate_user_prompt(
            self, 
            json_schema: str,
            text_response: str,
            error_message: str = "not provided",
        ) -> str:
        return self.user_prompt_template.format(
            json_schema=json_schema,
            text_response=text_response,
            error_message=error_message
        )

    def run(
            self, 
            text_response: str, 
            json_schema: BaseModel, 
            error_message: str = "not provided",
            model_id: str = 'qwen3:8b'
        ) -> dict:
        user_prompt = self._generate_user_prompt(
            json_schema.model_json_schema(), 
            text_response, 
            error_message
        )

        response: ChatResponse = chat(
            model=model_id,
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
        )

        return json.loads(response.message.content)
