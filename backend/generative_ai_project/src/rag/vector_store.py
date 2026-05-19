"""
Vector Store — Weaviate integration for semantic + hybrid search.

Uses Weaviate v4 Python client with gRPC. Vectorizer set to 'none'
since we provide our own BAAI/bge-m3 embeddings (multilingual).
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("rag.vector_store")

_COLLECTION_NAME = "ServiceProvider"


class WeaviateVectorStore:
    """Weaviate-based vector store for provider embeddings."""

    def __init__(
        self,
        url: Optional[str] = None,
        grpc_host: Optional[str] = None,
        grpc_port: Optional[int] = None,
        collection_name: str = _COLLECTION_NAME,
    ):
        import weaviate

        self.url = url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        _grpc_url = os.getenv("WEAVIATE_GRPC_URL", "localhost:50051")
        _grpc_host = grpc_host or _grpc_url.split(":")[0]
        _grpc_port = grpc_port or int(_grpc_url.split(":")[-1])

        self.client = weaviate.connect_to_local(
            host=self.url.replace("http://", "").replace("https://", "").split(":")[0],
            port=int(self.url.split(":")[-1]) if ":" in self.url.split("//")[-1] else 8080,
            grpc_port=_grpc_port,
        )
        self.collection_name = collection_name
        self._ensure_collection()
        logger.info(f"WeaviateVectorStore initialized: collection='{collection_name}'")

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        import weaviate.classes.config as wc

        if not self.client.collections.exists(self.collection_name):
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=wc.Configure.Vectorizer.none(),
                properties=[
                    wc.Property(name="provider_id", data_type=wc.DataType.INT),
                    wc.Property(name="provider_name", data_type=wc.DataType.TEXT),
                    wc.Property(name="category", data_type=wc.DataType.TEXT),
                    wc.Property(name="city", data_type=wc.DataType.TEXT),
                    wc.Property(name="area", data_type=wc.DataType.TEXT),
                    wc.Property(name="full_location", data_type=wc.DataType.TEXT),
                    wc.Property(name="latitude", data_type=wc.DataType.NUMBER),
                    wc.Property(name="longitude", data_type=wc.DataType.NUMBER),
                    wc.Property(name="rating", data_type=wc.DataType.NUMBER),
                    wc.Property(name="availability", data_type=wc.DataType.TEXT),
                    wc.Property(name="experience_years", data_type=wc.DataType.INT),
                    wc.Property(name="completed_jobs", data_type=wc.DataType.INT),
                    wc.Property(name="response_time_minutes", data_type=wc.DataType.INT),
                    wc.Property(name="price_range", data_type=wc.DataType.TEXT),
                    wc.Property(name="verified_provider", data_type=wc.DataType.BOOL),
                    wc.Property(name="languages_supported", data_type=wc.DataType.TEXT),
                    wc.Property(name="phone_number", data_type=wc.DataType.TEXT),
                    wc.Property(name="email", data_type=wc.DataType.TEXT),
                    wc.Property(name="chunk_text", data_type=wc.DataType.TEXT),
                ],
            )
            logger.info(f"Created Weaviate collection: {self.collection_name}")

    @property
    def count(self) -> int:
        """Number of objects in the collection."""
        try:
            collection = self.client.collections.get(self.collection_name)
            result = collection.aggregate.over_all(total_count=True)
            return result.total_count or 0
        except Exception:
            return 0

    def add_documents(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        batch_size: int = 200,
    ):
        """Batch-insert documents with embeddings into Weaviate."""
        collection = self.client.collections.get(self.collection_name)
        total = len(ids)
        logger.info(f"Adding {total} documents to Weaviate...")

        with collection.batch.dynamic() as batch:
            for i in range(total):
                properties = {**metadatas[i], "chunk_text": texts[i]}
                batch.add_object(
                    properties=properties,
                    vector=embeddings[i],
                )
                if (i + 1) % 5000 == 0:
                    logger.info(f"  Inserted {i + 1}/{total}")

        logger.info(f"Weaviate now contains {self.count} documents")

    def _object_to_document(self, obj) -> dict:
        """Convert a Weaviate object into the app's document shape."""
        props = {k: v for k, v in obj.properties.items()}
        return {
            "id": str(props.get("provider_id", "")),
            "text": props.pop("chunk_text", ""),
            "metadata": props,
            "distance": obj.metadata.distance if obj.metadata else 0.0,
        }

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Vector similarity search with optional metadata filters."""
        import weaviate.classes.query as wq

        collection = self.client.collections.get(self.collection_name)

        # Build Weaviate filter
        weaviate_filter = self._build_filter(filters) if filters else None

        try:
            results = collection.query.near_vector(
                near_vector=query_embedding,
                limit=n_results,
                filters=weaviate_filter,
                return_metadata=wq.MetadataQuery(distance=True),
            )
        except Exception as e:
            logger.warning(f"Filtered search failed: {e}. Trying without filters.")
            results = collection.query.near_vector(
                near_vector=query_embedding,
                limit=n_results,
                return_metadata=wq.MetadataQuery(distance=True),
            )

        documents = [self._object_to_document(obj) for obj in results.objects]

        logger.info(f"Weaviate search returned {len(documents)} results")
        return documents

    def get_provider_by_id(self, provider_id: int) -> Optional[dict]:
        """Fetch a single provider by its stored provider_id."""
        import weaviate.classes.query as wq

        collection = self.client.collections.get(self.collection_name)
        results = collection.query.fetch_objects(
            filters=wq.Filter.by_property("provider_id").equal(provider_id),
            limit=1,
        )

        if not results.objects:
            return None

        return self._object_to_document(results.objects[0])

    def hybrid_search(
        self,
        query_embedding: list[float],
        category: Optional[str] = None,
        city: Optional[str] = None,
        area: Optional[str] = None,
        availability: Optional[list[str]] = None,
        price_range: Optional[str] = None,
        n_results: int = 20,
    ) -> list[dict]:
        """Hybrid search combining vector similarity with metadata filters."""
        filters = {}
        if category:
            filters["category"] = category
        if city:
            filters["city"] = city
        if area:
            filters["area"] = area
        if availability:
            filters["availability"] = availability
        if price_range:
            filters["price_range"] = price_range

        return self.search(
            query_embedding=query_embedding,
            n_results=n_results,
            filters=filters if filters else None,
        )

    def _build_filter(self, filters: dict):
        """Build a Weaviate filter from a dict of conditions."""
        import weaviate.classes.query as wq

        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                # OR filter for list values (e.g., availability)
                or_conditions = []
                for v in value:
                    or_conditions.append(wq.Filter.by_property(key).equal(v))
                if or_conditions:
                    combined = or_conditions[0]
                    for c in or_conditions[1:]:
                        combined = combined | c
                    conditions.append(combined)
            else:
                conditions.append(wq.Filter.by_property(key).equal(value))

        if not conditions:
            return None
        result = conditions[0]
        for c in conditions[1:]:
            result = result & c
        return result

    def delete_collection(self):
        """Delete the entire collection."""
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)
            logger.info(f"Deleted Weaviate collection: {self.collection_name}")

    def close(self):
        """Close the Weaviate client connection."""
        self.client.close()
