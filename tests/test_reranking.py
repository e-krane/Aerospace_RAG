"""
Reranking quality validation tests.

Validates:
- Relevance improvement over base retrieval
- Latency <200ms per query
- Quality consistency across query types
"""

import pytest
import time
from pathlib import Path

from src.reranking.jina_colbert_reranker import JinaColBERTReranker
from src.reranking.optimization import OptimizedReranker, RerankerCache, BatchReranker
from loguru import logger


# Test data: technical aerospace queries
TEST_QUERIES = [
    "What is the stress-strain relationship in beam bending?",
    "Explain the moment of inertia for rectangular cross-sections",
    "How do you calculate shear stress in aircraft wing structures?",
    "Describe the theory of plate buckling under compression",
]

# Mock documents (relevant and irrelevant)
MOCK_DOCUMENTS = {
    "beam_bending": [
        "Beam bending stress is given by σ = My/I where M is moment, y is distance from neutral axis, and I is moment of inertia.",
        "The stress-strain relationship in elastic bending follows Hooke's law with σ = Eε.",
        "Aircraft wing structures use composite materials to reduce weight.",
        "Computational fluid dynamics (CFD) is used to analyze airflow.",
        "The moment of inertia depends on the cross-sectional geometry.",
    ],
    "moment_of_inertia": [
        "For a rectangular cross-section, I = bh³/12 where b is width and h is height.",
        "The moment of inertia is a geometric property that resists bending.",
        "Shear stress is maximum at the neutral axis in beam bending.",
        "Aerospace structures often use aluminum alloys for strength-to-weight ratio.",
        "The parallel axis theorem states I = I_c + Ad².",
    ],
    "shear_stress": [
        "Shear stress in beams is τ = VQ/(Ib) where V is shear force, Q is first moment of area.",
        "Aircraft wing structures experience both bending and shear stresses.",
        "The shear flow in thin-walled sections is q = τt where t is thickness.",
        "Plate buckling is governed by Euler's buckling formula.",
        "Composite materials exhibit orthotropic shear behavior.",
    ],
    "plate_buckling": [
        "Plate buckling under compression follows the critical stress formula σ_cr = k(π²E)/(12(1-ν²))(t/b)².",
        "The buckling coefficient k depends on boundary conditions and aspect ratio.",
        "Aerospace structures use stiffeners to prevent buckling.",
        "Shear stress is not the primary factor in buckling analysis.",
        "The theory of plate buckling was developed by Bryan in 1891.",
    ],
}


class TestRerankerQuality:
    """Test reranking quality and relevance improvement."""

    @pytest.fixture
    def reranker(self):
        """Create reranker instance."""
        return JinaColBERTReranker()

    def test_relevance_ordering(self, reranker):
        """Test that reranker improves relevance ordering."""
        query = TEST_QUERIES[0]  # Beam bending query
        documents = MOCK_DOCUMENTS["beam_bending"]

        # Rerank
        reranked_indices = reranker.rerank(query=query, documents=documents, top_k=3)

        # Most relevant should be first two documents
        assert reranked_indices[0] in [0, 1], "Most relevant document should be ranked first"
        assert reranked_indices[1] in [0, 1], "Second most relevant should be in top 2"

        # Irrelevant documents (CFD, wing materials) should be ranked lower
        assert 2 not in reranked_indices[:2], "Irrelevant document should not be in top 2"
        assert 3 not in reranked_indices[:2], "Irrelevant document should not be in top 2"

        logger.info(f"Relevance test passed: {reranked_indices}")

    def test_equation_matching(self, reranker):
        """Test reranker handles equation-heavy content."""
        query = "moment of inertia formula for rectangular beam"
        documents = MOCK_DOCUMENTS["moment_of_inertia"]

        reranked_indices = reranker.rerank(query=query, documents=documents, top_k=3)

        # Document with I = bh³/12 formula should rank highly
        assert reranked_indices[0] in [0, 1], "Equation document should rank in top 2"

        logger.info(f"Equation matching test passed: {reranked_indices}")

    def test_technical_term_precision(self, reranker):
        """Test precise technical term matching."""
        query = "shear flow in thin-walled sections"
        documents = MOCK_DOCUMENTS["shear_stress"]

        reranked_indices = reranker.rerank(query=query, documents=documents, top_k=3)

        # Document with shear flow formula should rank first
        assert reranked_indices[0] == 2, "Shear flow document should rank first"

        logger.info(f"Technical term test passed: {reranked_indices}")


class TestRerankerPerformance:
    """Test reranker performance and latency."""

    @pytest.fixture
    def reranker(self):
        """Create reranker instance."""
        return JinaColBERTReranker()

    def test_latency_single_query(self, reranker):
        """Test single query latency <200ms."""
        query = TEST_QUERIES[0]
        documents = MOCK_DOCUMENTS["beam_bending"]

        start_time = time.time()
        reranker.rerank(query=query, documents=documents, top_k=3)
        latency = (time.time() - start_time) * 1000

        assert latency < 200, f"Latency {latency:.1f}ms exceeds 200ms target"

        logger.info(f"Latency test passed: {latency:.1f}ms")

    def test_batch_processing_speed(self, reranker):
        """Test batch processing improves throughput."""
        batch_reranker = BatchReranker(reranker=reranker, batch_size=4)

        queries = TEST_QUERIES
        documents_list = [
            MOCK_DOCUMENTS["beam_bending"],
            MOCK_DOCUMENTS["moment_of_inertia"],
            MOCK_DOCUMENTS["shear_stress"],
            MOCK_DOCUMENTS["plate_buckling"],
        ]

        start_time = time.time()
        results = batch_reranker.rerank_batch(
            queries=queries,
            documents_list=documents_list,
            top_k=3,
        )
        total_time = time.time() - start_time

        avg_latency = (total_time / len(queries)) * 1000

        assert len(results) == len(queries), "Should return result for each query"
        assert avg_latency < 200, f"Avg latency {avg_latency:.1f}ms exceeds 200ms"

        logger.info(f"Batch processing test passed: {avg_latency:.1f}ms avg")


class TestRerankerCaching:
    """Test reranker caching optimization."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create cache instance with temp directory."""
        return RerankerCache(cache_dir=tmp_path / "cache", max_size=100)

    def test_cache_hit(self, cache):
        """Test cache stores and retrieves results."""
        query = TEST_QUERIES[0]
        documents = MOCK_DOCUMENTS["beam_bending"]
        result = [0, 1, 2, 3, 4]

        # Store
        cache.put(query, documents, result)

        # Retrieve
        cached_result = cache.get(query, documents)

        assert cached_result == result, "Cached result should match original"
        assert cache.stats.hits == 1, "Should register cache hit"

        logger.info("Cache hit test passed")

    def test_cache_miss(self, cache):
        """Test cache miss on new query."""
        query = "new query"
        documents = ["doc1", "doc2"]

        result = cache.get(query, documents)

        assert result is None, "Should return None on cache miss"
        assert cache.stats.misses == 1, "Should register cache miss"

        logger.info("Cache miss test passed")

    def test_optimized_reranker_speedup(self, tmp_path):
        """Test optimized reranker with cache speedup."""
        base_reranker = JinaColBERTReranker()
        opt_reranker = OptimizedReranker(
            base_reranker=base_reranker,
            enable_cache=True,
            cache_dir=tmp_path / "cache",
        )

        query = TEST_QUERIES[0]
        documents = MOCK_DOCUMENTS["beam_bending"]

        # First call (cache miss)
        start1 = time.time()
        result1 = opt_reranker.rerank(query=query, documents=documents, top_k=3)
        time1 = time.time() - start1

        # Second call (cache hit)
        start2 = time.time()
        result2 = opt_reranker.rerank(query=query, documents=documents, top_k=3)
        time2 = time.time() - start2

        assert result1 == result2, "Cached result should match original"
        assert time2 < time1 * 0.1, "Cache hit should be >10x faster"

        logger.info(
            f"Speedup test passed: {time1*1000:.1f}ms → {time2*1000:.1f}ms "
            f"({time1/time2:.1f}x speedup)"
        )


class TestRerankerConsistency:
    """Test reranker consistency across query types."""

    @pytest.fixture
    def reranker(self):
        """Create reranker instance."""
        return JinaColBERTReranker()

    def test_factual_query_consistency(self, reranker):
        """Test factual query handling."""
        query = "moment of inertia rectangular cross-section"
        documents = MOCK_DOCUMENTS["moment_of_inertia"]

        result = reranker.rerank(query=query, documents=documents, top_k=3)

        # Should prioritize formula document
        assert result[0] in [0, 1], "Formula should rank in top 2"

        logger.info(f"Factual query test passed: {result}")

    def test_conceptual_query_consistency(self, reranker):
        """Test conceptual query handling."""
        query = "explain the theory behind plate buckling"
        documents = MOCK_DOCUMENTS["plate_buckling"]

        result = reranker.rerank(query=query, documents=documents, top_k=3)

        # Should prioritize theoretical explanation
        assert result[0] in [0, 1, 4], "Theory documents should rank high"

        logger.info(f"Conceptual query test passed: {result}")

    def test_procedural_query_consistency(self, reranker):
        """Test procedural query handling."""
        query = "how to calculate shear stress in beams"
        documents = MOCK_DOCUMENTS["shear_stress"]

        result = reranker.rerank(query=query, documents=documents, top_k=3)

        # Should prioritize formula document
        assert result[0] in [0, 2], "Formula should rank first"

        logger.info(f"Procedural query test passed: {result}")


if __name__ == "__main__":
    logger.add("logs/test_reranking.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print("RERANKING QUALITY VALIDATION TESTS")
    print("=" * 70)
    print("\nTest Coverage:")
    print("  1. Relevance ordering improvement")
    print("  2. Equation matching accuracy")
    print("  3. Technical term precision")
    print("  4. Latency <200ms per query")
    print("  5. Batch processing speedup")
    print("  6. Cache hit/miss behavior")
    print("  7. Optimized reranker >10x speedup")
    print("  8. Consistency across query types")
    print("\nRun with: pytest tests/test_reranking.py -v")
    print("=" * 70)
