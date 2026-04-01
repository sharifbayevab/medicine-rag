import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models


class FaceVectorStore:
    def __init__(
        self,
        db_path: str = "data/faces/qdrant",
        collection_name: str = "people_faces",
        vector_size: int | None = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(path=str(self.db_path))

    def _collection_exists(self) -> bool:
        try:
            collections = self.client.get_collections().collections
        except Exception:
            return False
        return any(collection.name == self.collection_name for collection in collections)

    @staticmethod
    def _normalize(embedding: list[float]) -> list[float]:
        arr = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr.tolist()
        return (arr / norm).tolist()

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        if any(collection.name == self.collection_name for collection in collections):
            info = self.client.get_collection(self.collection_name)
            config_size = info.config.params.vectors.size
            if config_size != vector_size:
                self.client.delete_collection(self.collection_name)
            else:
                return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def identify(self, embedding: list[float], threshold: float = 0.62) -> dict | None:
        if not embedding:
            return None

        normalized = self._normalize(embedding)
        self._ensure_collection(len(normalized))

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            return None

        comparisons = []
        best_record = None
        best_score = -1.0

        for record in records:
            vector = record.vector
            if isinstance(vector, dict):
                vector = next(iter(vector.values()))
            if vector is None:
                continue

            score = self._cosine_similarity(normalized, vector)
            payload = record.payload or {}
            comparison = {
                "person_id": payload.get("person_id"),
                "first_name": payload.get("first_name", ""),
                "last_name": payload.get("last_name", ""),
                "full_name": f'{payload.get("first_name", "")} {payload.get("last_name", "")}'.strip(),
                "score": round(score, 4),
                "metadata": payload.get("metadata", {}),
            }
            comparisons.append(comparison)

            if score > best_score:
                best_score = score
                best_record = record

        if best_record is None:
            return None

        payload = best_record.payload or {}
        result = {
            "person_id": payload.get("person_id"),
            "first_name": payload.get("first_name", ""),
            "last_name": payload.get("last_name", ""),
            "full_name": f'{payload.get("first_name", "")} {payload.get("last_name", "")}'.strip(),
            "score": round(float(best_score), 4),
            "snapshots": payload.get("snapshots", []),
            "metadata": payload.get("metadata", {}),
            "matched": float(best_score) >= threshold,
            "comparisons": sorted(comparisons, key=lambda item: item["score"], reverse=True),
        }
        return result

    def register(
        self,
        embedding: list[float],
        first_name: str,
        last_name: str = "",
        snapshot_path: str | None = None,
        metadata: dict | None = None,
        person_id: str | None = None,
    ) -> dict:
        normalized = self._normalize(embedding)
        self._ensure_collection(len(normalized))

        person_id = person_id or str(uuid.uuid4())
        point_id = str(uuid.uuid4())
        payload = {
            "person_id": person_id,
            "first_name": first_name.strip(),
            "last_name": last_name.strip(),
            "snapshots": [snapshot_path] if snapshot_path else [],
            "metadata": metadata or {},
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=normalized,
                    payload=payload,
                )
            ],
        )

        return {
            "person_id": payload["person_id"],
            "first_name": payload["first_name"],
            "last_name": payload["last_name"],
            "full_name": f'{payload["first_name"]} {payload["last_name"]}'.strip(),
            "metadata": payload["metadata"],
        }

    def add_snapshot(self, person_id: str, snapshot_path: str) -> None:
        if not snapshot_path:
            return
        if not self._collection_exists():
            return

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="person_id",
                        match=models.MatchValue(value=person_id),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            return

        point = records[0]
        payload = point.payload or {}
        snapshots = payload.get("snapshots", [])
        if snapshot_path in snapshots:
            return

        snapshots.append(snapshot_path)
        payload["snapshots"] = snapshots

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=payload,
                )
            ],
        )

    def get_person(self, person_id: str) -> dict | None:
        if not self._collection_exists():
            return None

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="person_id",
                        match=models.MatchValue(value=person_id),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not records:
            return None

        payload = records[0].payload or {}
        return {
            "person_id": payload.get("person_id"),
            "first_name": payload.get("first_name", ""),
            "last_name": payload.get("last_name", ""),
            "full_name": f'{payload.get("first_name", "")} {payload.get("last_name", "")}'.strip(),
            "snapshots": payload.get("snapshots", []),
            "metadata": payload.get("metadata", {}),
        }
