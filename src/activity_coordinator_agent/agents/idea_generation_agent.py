from ..models.idea_generation_model import ActivityIdea
from ..models.context_model import Context
from ollama import chat, ChatResponse
import json
import numpy as np
import asyncio
from ollama import AsyncClient


SYSTEM_PROMPT = f"""
You are a creative assistant that generates engaging care-home activity ideas based on provided context.
You will receive context information about the residents and activity requirements in the format of JSON with following schema:
```
{json.dumps(Context.model_json_schema(), indent=2)}
```

Your task is to generate one creative idea in JSON format that conforms to the schema specified below:
```
{json.dumps(ActivityIdea.model_json_schema(), indent=2)}
```

You must strictly adhere to the following guidelines when extracting context and only include json only
"""

class IdeaGenerationAgent:
    def __init__(
            self, 
            system_prompt: str | None = None, 
            n_ideas: int = 3, 
            random_seed: int = 42,
            max_concurrency: int = 3
        ):
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        self.system_prompt = system_prompt
        self.n_ideas = n_ideas
        self.random_seed = random_seed
        self.temperature_list = self._generate_temperature_list()
        self.top_p = self._generate_top_p_list()
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def _generate_temperature_list(self):
        np.random.seed(self.random_seed)
        return np.random.uniform(0.7, 0.8, size=self.n_ideas).tolist()

    def _generate_top_p_list(self):
        np.random.seed(self.random_seed)
        return np.random.uniform(0.9, 1.0, size=self.n_ideas).tolist()

    async def _generate_idea(
            self,
            context: Context,
            model_id: str,
            temperature: float,
            top_p: float
        ) -> ActivityIdea:

        response: ChatResponse = await AsyncClient().chat(
            model=model_id,
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': context.model_dump_json(indent=2)},
            ],
            options={
                "temperature": temperature,
                "top_p": top_p
            }
        )

        try:
            response_json = json.loads(response.message.content)
            activity_ideas = ActivityIdea(**response_json)
        except Exception as e:
            from .json_correction_agent import JsonCorrectionAgent

            json_correction_agent = JsonCorrectionAgent()
            corrected_json = json_correction_agent.run(
                text_response=response.message.content,
                json_schema=ActivityIdea,
                error_message=str(e),
            )

            activity_ideas = ActivityIdea(**corrected_json)

        return activity_ideas


    async def run_async(self, context: Context, model_id: str = 'qwen3:8b') -> list[ActivityIdea]:     
        # Run all idea generations concurrently
        tasks = [
            self._generate_idea(
                context=context, 
                model_id=model_id, 
                temperature=temp, 
                top_p=top
            )
            for temp, top in zip(self.temperature_list, self.top_p)
        ]

        return await asyncio.gather(*tasks)

    def run(self, context: Context, model_id: str = 'qwen3:8b') -> list[ActivityIdea]:
        return asyncio.run(self.run_async(context=context, model_id=model_id))