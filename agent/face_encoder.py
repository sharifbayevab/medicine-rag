import base64
import io
import os
from pathlib import Path

from PIL import Image

try:
    from google import genai
except ImportError as exc:
    raise RuntimeError(
        "google-genai is not installed. Install it before using Gemini face embeddings."
    ) from exc


class FaceEncoder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("FACE_EMBED_MODEL", "gemini-embedding-2-preview")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini embeddings")
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _read_image(path: str) -> Image.Image:
        image = Image.open(str(Path(path))).convert("RGB")
        return image

    @staticmethod
    def _decode_base64_image(image_data: str) -> Image.Image:
        if "," in image_data:
            _, image_data = image_data.split(",", 1)
        binary = base64.b64decode(image_data)
        return Image.open(io.BytesIO(binary)).convert("RGB")

    def _embed_image(self, image: Image.Image) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=image,
        )
        if not response.embeddings:
            raise ValueError("Gemini did not return embeddings")
        values = response.embeddings[0].values
        if not values:
            raise ValueError("Gemini returned empty embedding")
        return list(values)

    def extract_embedding_from_path(self, path: str) -> list[float]:
        image = self._read_image(path)
        return self._embed_image(image)

    def extract_embedding_from_base64(self, image_data: str) -> list[float]:
        image = self._decode_base64_image(image_data)
        return self._embed_image(image)
