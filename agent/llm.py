import os
import json
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

    async def extract_person_name(self, text: str) -> dict | None:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Извлеки имя и фамилию человека из фразы. "
                        "Верни только JSON вида "
                        '{"first_name":"...","last_name":"...","is_confident":true}. '
                        "Если фамилии нет, верни пустую строку. "
                        "Если не уверен, верни is_confident=false."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except Exception:
            return None
        if not data.get("first_name"):
            return None
        return data
