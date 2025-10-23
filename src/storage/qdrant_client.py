"""
Qdrant vector database client for RAG system.

Features:
- Collection management with HNSW indexing
- Dense + sparse vector support (hybrid search)
- Binary quantization with int8 rescoring
- Metadata filtering and payload schema
- CRUD operations with error handling
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        HnswConfigDiff,
        OptimizersConfigDiff,
        QuantizationConfig,
        ScalarQuantization,
        ScalarType,
        QuantizationSearchParams,
        SparseVectorParams,
        SparseIndexParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        Range,
    )
except ImportError as e:
    raise ImportError(
        "qdrant-client required. Install with: pip install qdrant-client"
    ) from e

import numpy as np
from loguru import logger


@dataclass
class CollectionConfig:
    """Configuration for Qdrant collection."""

    name: str
    vector_size: int = 256  # Matryoshka compressed size
    distance: Distance = Distance.COSINE
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    full_scan_threshold: int = 10000
    enable_sparse_vectors: bool = True
    enable_quantization: bool = True
    on_disk: bool = False


class AerospaceQdrantClient:
    """
    Qdrant client for Aerospace RAG system.

    Features:
    - HNSW indexing for fast ANN search
    - Hybrid search (dense + sparse vectors)
    - Binary quantization with int8 rescoring
    - Metadata filtering support
    - Batch operations
    """

    # Collection names
    SEMANTIC_COLLECTION = "aerospace_semantic"
    EQUATION_COLLECTION = "aerospace_equations"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        """
        Initialize Qdrant client.

        Args:
            host: Qdrant host
            port: Qdrant port
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            timeout=timeout,
        )

        logger.info(f"Connected to Qdrant at {host}:{port}")

    def create_semantic_collection(
        self,
        config: Optional[CollectionConfig] = None,
        recreate: bool = False,
    ) -> bool:
        """
        Create collection for semantic search.

        Args:
            config: Collection configuration
            recreate: If True, delete existing collection first

        Returns:
            True if created successfully
        """
        if config is None:
            config = CollectionConfig(name=self.SEMANTIC_COLLECTION)

        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == config.name for c in collections)

        if exists:
            if recreate:
                logger.warning(f"Deleting existing collection: {config.name}")
                self.client.delete_collection(config.name)
            else:
                logger.info(f"Collection {config.name} already exists")
                return False

        # Create collection with dense vectors
        logger.info(f"Creating semantic collection: {config.name}")

        # Dense vector configuration
        vectors_config = VectorParams(
            size=config.vector_size,
            distance=config.distance,
            on_disk=config.on_disk,
            hnsw_config=HnswConfigDiff(
                m=config.hnsw_m,
                ef_construct=config.hnsw_ef_construct,
                full_scan_threshold=config.full_scan_threshold,
            ),
            quantization_config=(
                QuantizationConfig(
                    scalar=ScalarQuantization(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                )
                if config.enable_quantization
                else None
            ),
        )

        # Sparse vector configuration for BM25
        sparse_vectors_config = None
        if config.enable_sparse_vectors:
            sparse_vectors_config = {
                "bm25": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            }

        # Create collection
        self.client.create_collection(
            collection_name=config.name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2,
                indexing_threshold=10000,
            ),
        )

        # Create payload indexes for filtering
        self._create_payload_indexes(config.name)

        logger.info(
            f"Created collection {config.name}: "
            f"vectors={config.vector_size}D, "
            f"hnsw(m={config.hnsw_m}, ef={config.hnsw_ef_construct}), "
            f"sparse={config.enable_sparse_vectors}, "
            f"quantization={config.enable_quantization}"
        )

        return True

    def create_equation_collection(
        self,
        recreate: bool = False,
    ) -> bool:
        """
        Create collection for equation-specific search.

        Equation collection schema:
        - equation_id: Unique equation identifier
        - latex_source: Raw LaTeX string
        - normalized_form: Normalized LaTeX (for exact matching)
        - context_chunk_id: Link to parent semantic chunk
        - embedding: Dense vector for semantic search

        Args:
            recreate: If True, delete existing collection first

        Returns:
            True if created successfully
        """
        config = CollectionConfig(
            name=self.EQUATION_COLLECTION,
            vector_size=256,
            enable_sparse_vectors=False,  # Equations use dense vectors only
            enable_quantization=True,
        )

        success = self.create_semantic_collection(config=config, recreate=recreate)

        if success:
            # Create additional indexes for equation-specific fields
            self._create_equation_indexes()

        return success

    def _create_equation_indexes(self):
        """
        Create indexes for equation collection fields.

        Indexes:
        - equation_id (keyword)
        - latex_source (text)
        - normalized_form (keyword for exact matching)
        - context_chunk_id (keyword for cross-linking)
        """
        collection_name = self.EQUATION_COLLECTION

        # Equation ID index
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="equation_id",
            field_schema="keyword",
        )

        # LaTeX source index (text search)
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="latex_source",
            field_schema="text",
        )

        # Normalized form index (exact matching)
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="normalized_form",
            field_schema="keyword",
        )

        # Context chunk ID index (cross-linking)
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="context_chunk_id",
            field_schema="keyword",
        )

        logger.info(f"Created equation-specific indexes for {collection_name}")

    def upsert_equation(
        self,
        equation_id: str,
        latex_source: str,
        normalized_form: str,
        context_chunk_id: str,
        embedding: np.ndarray,
        additional_metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Insert or update an equation in the equation collection.

        Args:
            equation_id: Unique equation identifier
            latex_source: Raw LaTeX string
            normalized_form: Normalized LaTeX for exact matching
            context_chunk_id: ID of parent chunk in semantic collection
            embedding: Equation embedding vector
            additional_metadata: Optional additional metadata

        Returns:
            True if successful
        """
        payload = {
            "equation_id": equation_id,
            "latex_source": latex_source,
            "normalized_form": normalized_form,
            "context_chunk_id": context_chunk_id,
        }

        # Add additional metadata
        if additional_metadata:
            payload.update(additional_metadata)

        point = {
            "id": equation_id,
            "vector": embedding,
            "payload": payload,
        }

        try:
            self.upsert_points(
                collection_name=self.EQUATION_COLLECTION,
                points=[point],
            )
            return True

        except Exception as e:
            logger.error(f"Failed to upsert equation {equation_id}: {e}")
            return False

    def search_equations_by_latex(
        self,
        latex_query: str,
        limit: int = 10,
        exact_match: bool = False,
    ) -> List[Dict]:
        """
        Search equations by LaTeX string.

        Args:
            latex_query: LaTeX query string
            limit: Maximum results
            exact_match: If True, use normalized_form for exact matching

        Returns:
            List of matching equations with scores
        """
        from qdrant_client.models import FieldCondition, MatchText

        if exact_match:
            # Exact match on normalized form
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="normalized_form",
                        match=MatchValue(value=latex_query),
                    )
                ]
            )

            # Scroll through matches (no vector search needed)
            results, _ = self.scroll_points(
                collection_name=self.EQUATION_COLLECTION,
                limit=limit,
                with_payload=True,
            )

            # Filter by normalized_form
            filtered = [
                r for r in results
                if r.get("payload", {}).get("normalized_form") == latex_query
            ]

            return filtered[:limit]

        else:
            # Text search on latex_source
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="latex_source",
                        match=MatchText(text=latex_query),
                    )
                ]
            )

            # Need a dummy vector for search (filter-only search)
            dummy_vector = np.zeros(256)

            results = self.search(
                collection_name=self.EQUATION_COLLECTION,
                query_vector=dummy_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )

            return results

    def get_chunk_equations(
        self,
        context_chunk_id: str,
    ) -> List[Dict]:
        """
        Get all equations linked to a specific chunk.

        Args:
            context_chunk_id: Chunk ID from semantic collection

        Returns:
            List of equations in the chunk
        """
        from qdrant_client.models import FieldCondition, MatchValue

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="context_chunk_id",
                    match=MatchValue(value=context_chunk_id),
                )
            ]
        )

        # Scroll through matches
        results, _ = self.scroll_points(
            collection_name=self.EQUATION_COLLECTION,
            limit=100,  # Assume max 100 equations per chunk
            with_payload=True,
        )

        # Filter by context_chunk_id
        filtered = [
            r for r in results
            if r.get("payload", {}).get("context_chunk_id") == context_chunk_id
        ]

        return filtered

    def get_equation_context(
        self,
        equation_id: str,
    ) -> Optional[Dict]:
        """
        Get the parent chunk context for an equation.

        Args:
            equation_id: Equation ID

        Returns:
            Parent chunk from semantic collection or None
        """
        # Get equation point
        equation = self.get_point(
            collection_name=self.EQUATION_COLLECTION,
            point_id=equation_id,
            with_payload=True,
        )

        if not equation:
            return None

        # Get context chunk ID
        context_chunk_id = equation["payload"].get("context_chunk_id")
        if not context_chunk_id:
            return None

        # Get context chunk from semantic collection
        context_chunk = self.get_point(
            collection_name=self.SEMANTIC_COLLECTION,
            point_id=context_chunk_id,
            with_payload=True,
        )

        return context_chunk

    def _create_payload_indexes(self, collection_name: str):
        """
        Create indexes on payload fields for efficient filtering.

        Indexes:
        - document_id (keyword)
        - section_path (text)
        - chunk_type (keyword)
        - has_equations (boolean)
        - page_number (integer)
        """
        # Document ID index (keyword)
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema="keyword",
        )

        # Section path index (text)
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="section_path",
            field_schema="text",
        )

        # Chunk type index (keyword)
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="chunk_type",
            field_schema="keyword",
        )

        # Has equations index (keyword for boolean)
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="has_equations",
            field_schema="keyword",
        )

        # Page number index (integer)
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="page_number",
            field_schema="integer",
        )

        logger.info(f"Created payload indexes for {collection_name}")

    def upsert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Insert or update points in collection.

        Args:
            collection_name: Collection name
            points: List of point dictionaries with keys:
                - id: Point ID (optional, UUID generated if missing)
                - vector: Dense vector (required)
                - sparse_vector: Sparse vector dict (optional)
                - payload: Metadata dict (required)
            batch_size: Batch size for upload

        Returns:
            Number of points upserted
        """
        if not points:
            return 0

        # Convert to PointStruct objects
        qdrant_points = []

        for point in points:
            # Generate ID if missing
            point_id = point.get("id") or str(uuid.uuid4())

            # Dense vector (required)
            vector = point.get("vector")
            if vector is None:
                logger.warning(f"Point {point_id} missing vector, skipping")
                continue

            # Convert numpy arrays to lists
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()

            # Sparse vector (optional)
            sparse_vector = point.get("sparse_vector")
            if sparse_vector and isinstance(sparse_vector.get("values"), np.ndarray):
                sparse_vector["values"] = sparse_vector["values"].tolist()

            # Create PointStruct
            qdrant_point = PointStruct(
                id=point_id,
                vector=vector if not sparse_vector else {"dense": vector, "bm25": sparse_vector},
                payload=point.get("payload", {}),
            )

            qdrant_points.append(qdrant_point)

        # Batch upsert
        total_upserted = 0

        for i in range(0, len(qdrant_points), batch_size):
            batch = qdrant_points[i : i + batch_size]

            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )
                total_upserted += len(batch)

                if (i + batch_size) % 1000 == 0:
                    logger.info(f"Upserted {i + batch_size}/{len(qdrant_points)} points")

            except Exception as e:
                logger.error(f"Failed to upsert batch {i//batch_size}: {e}")

        logger.info(f"Upserted {total_upserted} points to {collection_name}")

        return total_upserted

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        score_threshold: Optional[float] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Dict]:
        """
        Search collection with dense vector.

        Args:
            collection_name: Collection name
            query_vector: Query vector
            limit: Maximum results to return
            query_filter: Optional Qdrant filter
            score_threshold: Minimum score threshold
            with_payload: Include payload in results
            with_vectors: Include vectors in results

        Returns:
            List of search results with scores and payloads
        """
        # Convert numpy to list
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        # Search
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

        # Convert to dicts
        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload if with_payload else None,
                "vector": r.vector if with_vectors else None,
            }
            for r in results
        ]

    def search_batch(
        self,
        collection_name: str,
        query_vectors: List[np.ndarray],
        limit: int = 10,
        query_filter: Optional[Filter] = None,
    ) -> List[List[Dict]]:
        """
        Batch search with multiple query vectors.

        Args:
            collection_name: Collection name
            query_vectors: List of query vectors
            limit: Maximum results per query
            query_filter: Optional filter

        Returns:
            List of result lists (one per query)
        """
        # Convert numpy arrays
        query_vectors = [
            v.tolist() if isinstance(v, np.ndarray) else v
            for v in query_vectors
        ]

        # Batch search
        results = self.client.search_batch(
            collection_name=collection_name,
            requests=[
                {
                    "vector": qv,
                    "limit": limit,
                    "filter": query_filter,
                    "with_payload": True,
                }
                for qv in query_vectors
            ],
        )

        # Convert to dicts
        return [
            [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload,
                }
                for r in batch_results
            ]
            for batch_results in results
        ]

    def get_point(
        self,
        collection_name: str,
        point_id: str,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Optional[Dict]:
        """
        Get a single point by ID.

        Args:
            collection_name: Collection name
            point_id: Point ID
            with_payload: Include payload
            with_vectors: Include vectors

        Returns:
            Point dict or None if not found
        """
        try:
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

            if result:
                point = result[0]
                return {
                    "id": point.id,
                    "payload": point.payload if with_payload else None,
                    "vector": point.vector if with_vectors else None,
                }

        except Exception as e:
            logger.error(f"Failed to get point {point_id}: {e}")

        return None

    def delete_points(
        self,
        collection_name: str,
        point_ids: List[str],
    ) -> bool:
        """
        Delete points by IDs.

        Args:
            collection_name: Collection name
            point_ids: List of point IDs to delete

        Returns:
            True if successful
        """
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids,
            )

            logger.info(f"Deleted {len(point_ids)} points from {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete points: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> Dict:
        """
        Get collection information.

        Args:
            collection_name: Collection name

        Returns:
            Collection info dict
        """
        try:
            info = self.client.get_collection(collection_name)

            return {
                "name": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else None,
                "vector_size": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else None,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def scroll_points(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[str] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Scroll through points in collection.

        Args:
            collection_name: Collection name
            limit: Maximum points to return
            offset: Scroll offset (point ID)
            with_payload: Include payload
            with_vectors: Include vectors

        Returns:
            Tuple of (points list, next_offset)
        """
        try:
            result = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

            points, next_offset = result

            return (
                [
                    {
                        "id": p.id,
                        "payload": p.payload if with_payload else None,
                        "vector": p.vector if with_vectors else None,
                    }
                    for p in points
                ],
                next_offset,
            )

        except Exception as e:
            logger.error(f"Failed to scroll points: {e}")
            return [], None

    @staticmethod
    def build_metadata_filter(
        document_id: Optional[str] = None,
        section_path: Optional[str] = None,
        chunk_type: Optional[str] = None,
        has_equations: Optional[bool] = None,
        page_number_min: Optional[int] = None,
        page_number_max: Optional[int] = None,
    ) -> Optional[Filter]:
        """
        Build a metadata filter for search queries.

        Args:
            document_id: Filter by document ID
            section_path: Filter by section path (text search)
            chunk_type: Filter by chunk type (text, equation, figure, mixed)
            has_equations: Filter by presence of equations
            page_number_min: Minimum page number
            page_number_max: Maximum page number

        Returns:
            Qdrant Filter object or None if no filters specified
        """
        from qdrant_client.models import FieldCondition, MatchText

        conditions = []

        # Document ID filter (exact match)
        if document_id:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            )

        # Section path filter (text search)
        if section_path:
            conditions.append(
                FieldCondition(
                    key="section_path",
                    match=MatchText(text=section_path),
                )
            )

        # Chunk type filter (exact match)
        if chunk_type:
            conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value=chunk_type),
                )
            )

        # Has equations filter (boolean)
        if has_equations is not None:
            conditions.append(
                FieldCondition(
                    key="has_equations",
                    match=MatchValue(value=has_equations),
                )
            )

        # Page number range filter
        if page_number_min is not None or page_number_max is not None:
            range_params = {}
            if page_number_min is not None:
                range_params["gte"] = page_number_min
            if page_number_max is not None:
                range_params["lte"] = page_number_max

            conditions.append(
                FieldCondition(
                    key="page_number",
                    range=Range(**range_params),
                )
            )

        # Return filter if conditions exist
        if conditions:
            return Filter(must=conditions)

        return None

    def search_with_filters(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        limit: int = 10,
        document_id: Optional[str] = None,
        section_path: Optional[str] = None,
        chunk_type: Optional[str] = None,
        has_equations: Optional[bool] = None,
        page_number_min: Optional[int] = None,
        page_number_max: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Search with metadata filters (convenience method).

        Args:
            collection_name: Collection name
            query_vector: Query vector
            limit: Maximum results
            document_id: Filter by document
            section_path: Filter by section
            chunk_type: Filter by type
            has_equations: Filter by equation presence
            page_number_min: Minimum page number
            page_number_max: Maximum page number
            score_threshold: Minimum score threshold

        Returns:
            Search results
        """
        # Build filter
        query_filter = self.build_metadata_filter(
            document_id=document_id,
            section_path=section_path,
            chunk_type=chunk_type,
            has_equations=has_equations,
            page_number_min=page_number_min,
            page_number_max=page_number_max,
        )

        # Search
        return self.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )


def create_qdrant_client(
    host: str = "localhost",
    port: int = 6333,
    api_key: Optional[str] = None,
) -> AerospaceQdrantClient:
    """
    Factory function to create Qdrant client.

    Args:
        host: Qdrant host
        port: Qdrant port
        api_key: Optional API key

    Returns:
        Configured AerospaceQdrantClient
    """
    return AerospaceQdrantClient(host=host, port=port, api_key=api_key)


if __name__ == "__main__":
    # Example usage and tests
    logger.add("logs/qdrant_client.log", rotation="10 MB")

    # Create client
    client = create_qdrant_client()

    # Create semantic collection
    client.create_semantic_collection(recreate=True)

    # Test CRUD operations
    print("\nTesting CRUD operations...")

    # Create test point
    test_vector = np.random.rand(256).astype(np.float32)
    test_point = {
        "id": "test_001",
        "vector": test_vector,
        "payload": {
            "content": "Test chunk content",
            "document_id": "test_doc",
            "section_path": ["Chapter 1", "Section 1.1"],
            "chunk_type": "text",
            "has_equations": False,
            "page_number": 1,
        },
    }

    # Upsert
    client.upsert_points(client.SEMANTIC_COLLECTION, [test_point])
    print("✓ Upsert successful")

    # Get point
    retrieved = client.get_point(client.SEMANTIC_COLLECTION, "test_001")
    print(f"✓ Retrieved point: {retrieved['id']}")

    # Search
    results = client.search(
        client.SEMANTIC_COLLECTION,
        test_vector,
        limit=5,
    )
    print(f"✓ Search returned {len(results)} results")

    # Get collection info
    info = client.get_collection_info(client.SEMANTIC_COLLECTION)
    print(f"✓ Collection info: {info['points_count']} points")

    # Delete
    client.delete_points(client.SEMANTIC_COLLECTION, ["test_001"])
    print("✓ Delete successful")

    print("\n✓ All CRUD tests passed")
