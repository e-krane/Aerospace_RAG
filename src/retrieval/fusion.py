"""
Reciprocal Rank Fusion (RRF) for combining BM25 and semantic search results.

RRF Algorithm:
score(d) = Σ 1/(k + rank_i(d))

where rank_i(d) is the rank of document d in ranker i, and k is a constant (typically 60).

Features:
- Combine multiple ranked lists
- Configurable α parameter for weighting
- Support for different fusion strategies
"""

from typing import List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass

from loguru import logger


@dataclass
class FusionResult:
    """Result after fusion."""

    id: str
    fused_score: float
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    semantic_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    payload: Optional[Dict] = None


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion for hybrid retrieval.

    Combines results from multiple retrieval methods using RRF algorithm.
    """

    def __init__(
        self,
        k: int = 60,
        alpha: float = 0.5,
    ):
        """
        Initialize RRF fusion.

        Args:
            k: RRF constant (default 60)
            alpha: Weighting parameter:
                   - α=0.7: favor BM25 (technical terminology)
                   - α=0.5: balanced (default)
                   - α=0.3: favor semantic (conceptual queries)
        """
        self.k = k
        self.alpha = alpha

        logger.info(f"RRF initialized: k={k}, α={alpha}")

    def fuse(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[FusionResult]:
        """
        Fuse semantic and BM25 results using RRF.

        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            top_k: Number of results to return (None = all)

        Returns:
            List of fused results sorted by RRF score
        """
        # Index results by ID
        all_docs = {}

        # Process semantic results
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result.get("id")
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "id": doc_id,
                    "semantic_score": result.get("score"),
                    "semantic_rank": rank,
                    "bm25_score": None,
                    "bm25_rank": None,
                    "payload": result.get("payload"),
                }

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result.get("id")
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "id": doc_id,
                    "semantic_score": None,
                    "semantic_rank": None,
                    "bm25_score": result.get("score"),
                    "bm25_rank": rank,
                    "payload": result.get("payload"),
                }
            else:
                all_docs[doc_id]["bm25_score"] = result.get("score")
                all_docs[doc_id]["bm25_rank"] = rank

        # Compute RRF scores
        fused_results = []

        for doc_id, doc_data in all_docs.items():
            # RRF formula: 1/(k + rank)
            semantic_rrf = 0.0
            if doc_data["semantic_rank"] is not None:
                semantic_rrf = 1.0 / (self.k + doc_data["semantic_rank"])

            bm25_rrf = 0.0
            if doc_data["bm25_rank"] is not None:
                bm25_rrf = 1.0 / (self.k + doc_data["bm25_rank"])

            # Weighted combination
            fused_score = self.alpha * semantic_rrf + (1 - self.alpha) * bm25_rrf

            fused_results.append(
                FusionResult(
                    id=doc_id,
                    fused_score=fused_score,
                    semantic_score=doc_data["semantic_score"],
                    bm25_score=doc_data["bm25_score"],
                    semantic_rank=doc_data["semantic_rank"],
                    bm25_rank=doc_data["bm25_rank"],
                    payload=doc_data["payload"],
                )
            )

        # Sort by fused score
        fused_results.sort(key=lambda x: x.fused_score, reverse=True)

        # Return top-k
        if top_k:
            fused_results = fused_results[:top_k]

        logger.info(
            f"RRF fusion: {len(semantic_results)} semantic + {len(bm25_results)} BM25 "
            f"-> {len(fused_results)} fused results"
        )

        return fused_results


def fuse_results(
    semantic_results: List[Dict],
    bm25_results: List[Dict],
    alpha: float = 0.5,
    top_k: int = 10,
) -> List[FusionResult]:
    """
    Convenience function for RRF fusion.

    Args:
        semantic_results: Semantic search results
        bm25_results: BM25 search results
        alpha: Weighting parameter (0.3-0.7)
        top_k: Number of results

    Returns:
        Fused results
    """
    fusion = ReciprocalRankFusion(alpha=alpha)
    return fusion.fuse(semantic_results, bm25_results, top_k=top_k)


if __name__ == "__main__":
    # Example usage
    logger.add("logs/fusion.log", rotation="10 MB")

    # Mock results
    semantic_results = [
        {"id": "doc_1", "score": 0.95},
        {"id": "doc_2", "score": 0.90},
        {"id": "doc_3", "score": 0.85},
        {"id": "doc_5", "score": 0.80},
    ]

    bm25_results = [
        {"id": "doc_2", "score": 45.5},
        {"id": "doc_4", "score": 42.3},
        {"id": "doc_1", "score": 40.1},
        {"id": "doc_6", "score": 38.2},
    ]

    # Test different α values
    alphas = [0.3, 0.5, 0.7]

    for alpha in alphas:
        print(f"\n=== α = {alpha} ===")
        fusion = ReciprocalRankFusion(alpha=alpha)
        results = fusion.fuse(semantic_results, bm25_results, top_k=5)

        for i, result in enumerate(results, 1):
            print(
                f"{i}. {result.id}: fused={result.fused_score:.4f} "
                f"(sem={result.semantic_score or 'N/A'}, bm25={result.bm25_score or 'N/A'})"
            )
