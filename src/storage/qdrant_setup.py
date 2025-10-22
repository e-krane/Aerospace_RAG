"""Qdrant client with collection setup for Aerospace RAG."""

from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PayloadSchemaType,
    QuantizationConfig,
    BinaryQuantization,
    ScalarQuantization,
    ScalarType,
    HnswConfigDiff,
)
from loguru import logger


class AerospaceQdrantClient:
    """Qdrant client for Aerospace RAG system."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
    ):
        """Initialize Qdrant client.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            api_key: Optional API key for authentication
        """
        self.client = QdrantClient(host=host, port=port, api_key=api_key)
        logger.info(f"Connected to Qdrant at {host}:{port}")

    def create_semantic_collection(
        self,
        collection_name: str = "aerospace_semantic",
        vector_size: int = 256,  # Matryoshka reduced dimension
        recreate: bool = False,
    ) -> None:
        """Create semantic collection with dense and sparse vectors.

        Args:
            collection_name: Name of the collection
            vector_size: Embedding dimension (256 for Matryoshka)
            recreate: If True, delete existing collection first
        """
        if recreate:
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except Exception as e:
                logger.debug(f"Collection {collection_name} did not exist: {e}")

        # Create collection with dense vectors
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=16,  # Number of edges per node
                    ef_construct=200,  # Quality of graph construction
                    full_scan_threshold=10000,  # Use exact search for small collections
                ),
                # Quantization will be added later during optimization phase
                # quantization_config=QuantizationConfig(
                #     scalar=ScalarQuantization(
                #         type=ScalarType.INT8,
                #         quantile=0.99,
                #         always_ram=True,
                #     ),
                # ),
            ),
            sparse_vectors_config={
                "text": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                ),
            },
        )

        # Create payload indexes for efficient filtering
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="chunk_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="has_equations",
            field_schema=PayloadSchemaType.BOOL,
        )
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="page_number",
            field_schema=PayloadSchemaType.INTEGER,
        )

        logger.info(f"Created collection: {collection_name} with {vector_size}D vectors")

    def test_connection(self) -> bool:
        """Test connection to Qdrant.

        Returns:
            True if connection successful
        """
        try:
            collections = self.client.get_collections()
            logger.info(f"Successfully connected. Collections: {len(collections.collections)}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False


if __name__ == "__main__":
    # Test connection and create collection
    client = AerospaceQdrantClient()

    if client.test_connection():
        print("✅ Qdrant connection successful")

        # Create semantic collection
        client.create_semantic_collection(recreate=True)
        print("✅ Semantic collection created")

        # Verify collection
        collections = client.client.get_collections()
        print(f"✅ Active collections: {[c.name for c in collections.collections]}")
    else:
        print("❌ Qdrant connection failed")
