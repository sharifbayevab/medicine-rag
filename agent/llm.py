import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    def __init__(self, api_key: str = None, model: str = "gpt-4.1"):
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    async def get_response_stream(self, messages: list):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )
        async for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
