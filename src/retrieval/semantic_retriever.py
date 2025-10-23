"""
Semantic vector search retriever using Qwen3 embeddings.

Features:
- Query embedding with Qwen3-8B
- Cosine similarity search in Qdrant
- Metadata filtering with automatic detection
- Query expansion for ambiguous terms
- Top-K retrieval with scores
"""

from typing import List, Dict, Optional
import re

try:
    import numpy as np
except ImportError as e:
    raise ImportError("numpy required. Install with: pip install numpy") from e

from loguru import logger

# Import dependencies
try:
    from ..embeddings.qwen3_embedder import Qwen3Embedder
    from ..storage.qdrant_client import AerospaceQdrantClient
except ImportError:
    from embeddings.qwen3_embedder import Qwen3Embedder
    from storage.qdrant_client import AerospaceQdrantClient


class SemanticRetriever:
    """
    Semantic retriever using dense vector embeddings.

    Features:
    - Qwen3-8B query embedding
    - Cosine similarity search
    - Automatic filter detection from query
    - Query expansion support
    """

    def __init__(
        self,
        embedder: Qwen3Embedder,
        qdrant_client: AerospaceQdrantClient,
    ):
        """
        Initialize semantic retriever.

        Args:
            embedder: Qwen3 embedder instance
            qdrant_client: Qdrant client instance
        """
        self.embedder = embedder
        self.client = qdrant_client

        logger.info("Semantic retriever initialized")

    def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        auto_filter: bool = True,
    ) -> List[Dict]:
        """
        Semantic search using query embedding.

        Args:
            query: Query string
            collection_name: Qdrant collection name
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            auto_filter: Automatically detect filters from query

        Returns:
            List of results with similarity scores
        """
        # Embed query
        query_embedding = self.embedder.embed(query)[0]

        # Detect filters from query
        filters = {}
        if auto_filter:
            filters = self._detect_filters(query)

        # Search
        results = self.client.search_with_filters(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            **filters,
        )

        logger.info(f"Semantic search: '{query}' -> {len(results)} results")

        return results

    def _detect_filters(self, query: str) -> Dict:
        """
        Automatically detect metadata filters from query.

        Patterns:
        - "in chapter X" -> document_id filter
        - "with equations" -> has_equations=True
        - "page X to Y" -> page_number range

        Args:
            query: Query string

        Returns:
            Dict of filter parameters
        """
        filters = {}

        # Chapter filter: "in chapter 10", "from chapter 5"
        chapter_match = re.search(r'(?:in|from)\s+chapter\s+(\d+)', query, re.IGNORECASE)
        if chapter_match:
            chapter_num = int(chapter_match.group(1))
            filters["document_id"] = f"chapter_{chapter_num:02d}"

        # Equation filter: "with equations", "containing formulas"
        if re.search(r'\b(with|containing|has)\s+(equation|formula)', query, re.IGNORECASE):
            filters["has_equations"] = True

        # Page range filter: "page 10 to 20", "pages 5-15"
        page_range_match = re.search(
            r'pages?\s+(\d+)\s*(?:to|-)\s*(\d+)',
            query,
            re.IGNORECASE
        )
        if page_range_match:
            filters["page_number_min"] = int(page_range_match.group(1))
            filters["page_number_max"] = int(page_range_match.group(2))

        if filters:
            logger.info(f"Auto-detected filters: {filters}")

        return filters


def semantic_search(
    query: str,
    embedder: Qwen3Embedder,
    qdrant_client: AerospaceQdrantClient,
    collection_name: str,
    top_k: int = 10,
) -> List[Dict]:
    """
    Convenience function for semantic search.

    Args:
        query: Query string
        embedder: Qwen3 embedder
        qdrant_client: Qdrant client
        collection_name: Collection name
        top_k: Number of results

    Returns:
        Search results
    """
    retriever = SemanticRetriever(embedder, qdrant_client)
    return retriever.search(query, collection_name, top_k=top_k)


if __name__ == "__main__":
    logger.add("logs/semantic_retriever.log", rotation="10 MB")

    # Test filter detection
    retriever = SemanticRetriever(None, None)

    test_queries = [
        "Euler buckling in chapter 10",
        "stress analysis with equations",
        "deflection formulas on page 50 to 60",
        "material properties from chapter 3 with equations",
    ]

    print("\nAutomatic Filter Detection:")
    for query in test_queries:
        filters = retriever._detect_filters(query)
        print(f"\nQuery: {query}")
        print(f"Filters: {filters}")
