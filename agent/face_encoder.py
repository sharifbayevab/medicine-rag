import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis


class FaceEncoder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("FACE_MODEL", "buffalo_s")
        self.model_root = os.getenv("FACE_MODEL_ROOT", "data/faces/models")
        self.det_size = int(os.getenv("FACE_DET_SIZE", "640"))
        provider = os.getenv("FACE_EXECUTION_PROVIDER", "CPUExecutionProvider")

        self.app = FaceAnalysis(
            name=self.model_name,
            root=self.model_root,
            providers=[provider],
        )
        self.app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))

    @staticmethod
    def _normalize_embedding(values: np.ndarray | list[float]) -> list[float]:
        arr = np.asarray(values, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr.tolist()
        return (arr / norm).tolist()

    @staticmethod
    def _decode_base64_image(image_data: str) -> np.ndarray:
        if "," in image_data:
            _, image_data = image_data.split(",", 1)
        binary = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(binary)).convert("RGB")
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _read_image(path: str) -> np.ndarray:
        image = Image.open(str(Path(path))).convert("RGB")
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _face_area(face) -> float:
        x1, y1, x2, y2 = face.bbox.astype(int)
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    @staticmethod
    def _crop_face(image: np.ndarray, bbox: np.ndarray, pad_ratio: float = 0.04) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        pad_x = int((x2 - x1) * pad_ratio)
        pad_y = int((y2 - y1) * pad_ratio)
        sx = max(0, x1 - pad_x)
        sy = max(0, y1 - pad_y)
        ex = min(w, x2 + pad_x)
        ey = min(h, y2 + pad_y)
        return image[sy:ey, sx:ex]

    @staticmethod
    def _encode_crop_base64(crop_bgr: np.ndarray) -> str | None:
        if crop_bgr.size == 0:
            return None
        ok, encoded = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return None
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def analyze_frame_base64(self, image_data: str, max_faces: int = 5) -> list[dict]:
        image = self._decode_base64_image(image_data)
        faces = self.app.get(image)
        if not faces:
            return []

        faces = sorted(faces, key=self._face_area, reverse=True)[:max_faces]
        results = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            crop = self._crop_face(image, face.bbox)
            crop_b64 = self._encode_crop_base64(crop)
            results.append(
                {
                    "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "embedding": self._normalize_embedding(face.normed_embedding),
                    "det_score": round(float(getattr(face, "det_score", 0.0)), 4),
                    "crop_base64": crop_b64,
                }
            )
        return results

    def extract_embedding_from_path(self, path: str) -> list[float]:
        image = self._read_image(path)
        faces = self.app.get(image)
        if not faces:
            raise ValueError("No face detected")
        face = max(faces, key=self._face_area)
        return self._normalize_embedding(face.normed_embedding)

    def extract_embedding_from_base64(self, image_data: str) -> list[float]:
        faces = self.analyze_frame_base64(image_data, max_faces=1)
        if not faces:
            raise ValueError("No face detected")
        return faces[0]["embedding"]
