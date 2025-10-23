"""
Two-stage retrieval pipeline with reranking.

Stage 1: Retrieve 100 candidates via hybrid search (BM25 + Semantic + RRF)
Stage 2: Rerank top candidates via Jina-ColBERT-v2
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import time

from loguru import logger

from src.retrieval.query_analyzer import QueryAnalyzer, QueryAnalysis
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import ReciprocalRankFusion
from src.reranking.jina_colbert_reranker import JinaColBERTReranker


@dataclass
class RetrievalResult:
    """Complete retrieval result with reranking."""

    query: str
    query_analysis: QueryAnalysis
    initial_candidates: int
    reranked_results: List[Dict]
    stage1_time: float
    stage2_time: float
    total_time: float


class TwoStageRetriever:
    """
    Two-stage retrieval pipeline.

    Stage 1: Hybrid retrieval (retrieve 100 candidates)
    Stage 2: Reranking (rerank to top 10)

    Performance target: <500ms total
    """

    def __init__(
        self,
        qdrant_client,
        collection_name: str = "aerospace_documents_semantic",
        stage1_k: int = 100,
        stage2_k: int = 10,
        reranker: Optional[JinaColBERTReranker] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
    ):
        """
        Initialize two-stage retriever.

        Args:
            qdrant_client: Qdrant client instance
            collection_name: Qdrant collection name
            stage1_k: Number of candidates from Stage 1 (default: 100)
            stage2_k: Number of final results from Stage 2 (default: 10)
            reranker: Optional reranker (creates if None)
            query_analyzer: Optional query analyzer (creates if None)
        """
        self.qdrant = qdrant_client
        self.collection_name = collection_name
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k

        # Initialize components
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.semantic_retriever = SemanticRetriever(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            query_analyzer=self.query_analyzer,
        )
        self.bm25_retriever = BM25Retriever()
        self.fusion = ReciprocalRankFusion()
        self.reranker = reranker or JinaColBERTReranker()

        logger.info(
            f"TwoStageRetriever initialized: Stage1={stage1_k}, Stage2={stage2_k}"
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None,
    ) -> RetrievalResult:
        """
        Execute two-stage retrieval.

        Args:
            query: User query string
            top_k: Number of final results (default: self.stage2_k)
            filters: Optional metadata filters

        Returns:
            RetrievalResult with reranked documents
        """
        start_time = time.time()

        if top_k is None:
            top_k = self.stage2_k

        # Analyze query
        query_analysis = self.query_analyzer.analyze(query)

        # Merge filters
        all_filters = {**query_analysis.filters, **(filters or {})}

        logger.info(
            f"Retrieving for query: '{query}' (type={query_analysis.query_type}, "
            f"alpha={query_analysis.alpha:.2f})"
        )

        # Stage 1: Hybrid retrieval
        stage1_start = time.time()
        candidates = self._stage1_retrieve(
            query=query,
            expanded_query=query_analysis.expanded_query,
            alpha=query_analysis.alpha,
            filters=all_filters,
        )
        stage1_time = time.time() - stage1_start

        logger.info(f"Stage 1 complete: {len(candidates)} candidates in {stage1_time:.3f}s")

        # Stage 2: Reranking
        stage2_start = time.time()
        reranked = self._stage2_rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )
        stage2_time = time.time() - stage2_start

        logger.info(f"Stage 2 complete: {len(reranked)} results in {stage2_time:.3f}s")

        total_time = time.time() - start_time

        return RetrievalResult(
            query=query,
            query_analysis=query_analysis,
            initial_candidates=len(candidates),
            reranked_results=reranked,
            stage1_time=stage1_time,
            stage2_time=stage2_time,
            total_time=total_time,
        )

    def _stage1_retrieve(
        self,
        query: str,
        expanded_query: str,
        alpha: float,
        filters: Dict,
    ) -> List[Dict]:
        """
        Stage 1: Hybrid retrieval via RRF.

        Args:
            query: Original query
            expanded_query: Query with term expansion
            alpha: Fusion weight
            filters: Metadata filters

        Returns:
            List of candidate documents
        """
        # Semantic search
        semantic_results = self.semantic_retriever.search(
            query=expanded_query,
            top_k=self.stage1_k,
            filters=filters,
        )

        # BM25 search
        bm25_results = self._bm25_search(
            query=query,
            top_k=self.stage1_k,
            filters=filters,
        )

        # Fuse results
        self.fusion.alpha = alpha
        fused_results = self.fusion.fuse(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            top_k=self.stage1_k,
        )

        return fused_results

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        filters: Dict,
    ) -> List[Dict]:
        """
        Execute BM25 search via Qdrant sparse vectors.

        Args:
            query: Query string
            top_k: Number of results
            filters: Metadata filters

        Returns:
            List of BM25 results
        """
        # Generate BM25 sparse vector
        sparse_vector = self.bm25_retriever.get_sparse_vector(query)

        # Build Qdrant filter
        qdrant_filter = self._build_qdrant_filter(filters)

        # Search
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=None,
            sparse_vector=sparse_vector,
            query_filter=qdrant_filter,
            limit=top_k,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": hit.payload,
            }
            for hit in results
        ]

    def _stage2_rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        """
        Stage 2: Rerank candidates with Jina-ColBERT-v2.

        Args:
            query: Query string
            candidates: Candidate documents from Stage 1
            top_k: Number of final results

        Returns:
            Reranked list of top_k documents
        """
        if not candidates:
            logger.warning("No candidates to rerank")
            return []

        # Extract texts
        texts = [c.get("text", "") for c in candidates]

        # Rerank
        reranked_indices = self.reranker.rerank(
            query=query,
            documents=texts,
            top_k=top_k,
        )

        # Return reranked candidates
        return [
            {
                **candidates[idx],
                "rerank_score": self.reranker.scores[i] if i < len(self.reranker.scores) else 0.0,
                "rerank_position": i + 1,
            }
            for i, idx in enumerate(reranked_indices[:top_k])
        ]

    def _build_qdrant_filter(self, filters: Dict):
        """
        Build Qdrant filter from dictionary.

        Args:
            filters: Filter dictionary

        Returns:
            Qdrant filter object
        """
        if not filters:
            return None

        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

        conditions = []

        for key, value in filters.items():
            if isinstance(value, dict):
                # Range filter (e.g., page_number)
                conditions.append(
                    FieldCondition(
                        key=key,
                        range=Range(**value),
                    )
                )
            elif isinstance(value, bool):
                # Boolean filter
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            else:
                # Exact match
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )

        return Filter(must=conditions) if conditions else None

    def retrieve_with_equations(
        self,
        query: str,
        top_k: int = 10,
    ) -> Dict:
        """
        Retrieve with automatic equation context enrichment.

        If results contain equations, fetch full equation context
        from equation collection.

        Args:
            query: User query
            top_k: Number of results

        Returns:
            Dict with results and equation context
        """
        # Standard retrieval
        result = self.retrieve(query=query, top_k=top_k)

        # Enrich with equation context
        enriched_results = []
        for doc in result.reranked_results:
            if doc.get("metadata", {}).get("has_equations"):
                # Fetch equation context
                chunk_id = doc.get("id")
                equations = self.qdrant.get_chunk_equations(chunk_id)
                doc["equations"] = equations

            enriched_results.append(doc)

        return {
            "query": query,
            "results": enriched_results,
            "retrieval_stats": {
                "stage1_candidates": result.initial_candidates,
                "stage2_results": len(result.reranked_results),
                "stage1_time": result.stage1_time,
                "stage2_time": result.stage2_time,
                "total_time": result.total_time,
            },
            "query_analysis": result.query_analysis,
        }


if __name__ == "__main__":
    logger.add("logs/two_stage_pipeline.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print("TWO-STAGE RETRIEVAL PIPELINE")
    print("=" * 70)
    print("\nStage 1: Hybrid retrieval (100 candidates)")
    print("  - Semantic search (Qwen3-8B)")
    print("  - BM25 keyword search")
    print("  - RRF fusion with dynamic alpha")
    print("\nStage 2: Reranking (top 10)")
    print("  - Jina-ColBERT-v2 (8192 token context)")
    print("  - <200ms latency target")
    print("\nTarget: <500ms total latency")
    print("=" * 70)
