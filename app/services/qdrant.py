from typing import List, Dict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantHelper:
    def __init__(self, url: str, collection_name: str, workspace_id: str = None):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.client = QdrantClient(url=url, prefer_grpc=False)
        self.base_collection_name = collection_name
        self.workspace_id = workspace_id
        self.collection_name = f"{workspace_id}_{collection_name}" if workspace_id else collection_name
        self.vector_size = 384
        self._init_collection()

    def _init_collection(self):
        try:
            collections = self.client.get_collections()
            names = [c.name for c in collections.collections]
            if self.collection_name not in names:
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
                )
        except Exception as e:
            raise ValueError(f"Error initializing collection: {e}")

    def add_vector(self, vector: np.ndarray, payload: Dict):
        if vector.ndim == 1:
            vector = vector.tolist()
        elif vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0].tolist()
        else:
            raise ValueError("Vector must be 1D or 2D with shape (1, N)")

        point_id = payload.get("ticket_id")
        
        if self.workspace_id:
            payload = {**payload, "workspace_id": self.workspace_id}

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(id=point_id, vector=vector, payload=payload)
                ]
            )
        except Exception as e:
            raise ValueError(f"Error adding vector: {e}")

    def search_similar(self, query_vector: np.ndarray, top_k: int = 5, score_threshold: float = 0.1) -> List[Dict]:
        if query_vector.ndim == 1:
            query_vector = query_vector.tolist()
        elif query_vector.ndim == 2 and query_vector.shape[0] == 1:
            query_vector = query_vector[0].tolist()
        else:
            raise ValueError("query_vector must be 1D or 2D with shape (1, N)")

        try:
            query_filter = None
            if self.workspace_id:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="workspace_id",
                            match=models.MatchValue(value=self.workspace_id)
                        )
                    ]
                )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                query_filter=query_filter
            )
            
            processed_results = [
                {
                    "ticket_id": point.payload.get("ticket_id"),
                    "question": point.payload.get("question"),
                    "dialog": point.payload.get("dialog"),
                    "score": point.score
                }
                for point in results
                if point.score >= score_threshold
            ]
            
            return processed_results
            
        except Exception as e:
            raise ValueError(f"Error searching vectors: {e}")
