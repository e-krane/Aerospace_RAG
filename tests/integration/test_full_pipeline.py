"""
Integration tests for complete RAG pipeline.

Tests the full end-to-end flow:
1. Configuration loading
2. Model initialization
3. Document indexing (if test document available)
4. Query processing
5. Performance benchmarks
6. Quality validation
"""

import pytest
from pathlib import Path
import time
from loguru import logger

from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.config import get_config


# Test configuration
TEST_QUERIES = [
    "What is the Euler buckling formula?",
    "Explain the relationship between stress and strain",
    "How do you calculate moment of inertia?",
]

EXPECTED_KEYWORDS = {
    "What is the Euler buckling formula?": ["euler", "buckling", "formula", "load", "column"],
    "Explain the relationship between stress and strain": ["stress", "strain", "modulus"],
    "How do you calculate moment of inertia?": ["moment", "inertia"],
}


@pytest.fixture(scope="module")
def config():
    """Load system configuration."""
    return get_config()


@pytest.fixture(scope="module")
def pipeline(config):
    """Initialize RAG pipeline once for all tests."""
    logger.info("Initializing RAG pipeline for integration tests...")
    start = time.time()

    pipeline = RAGPipeline(preload_models=True)

    elapsed = time.time() - start
    logger.info(f"Pipeline initialized in {elapsed:.2f}s")

    yield pipeline

    logger.info("Pipeline tests complete")


class TestConfiguration:
    """Test configuration loading."""

    def test_config_loads(self, config):
        """Configuration loads successfully."""
        assert config is not None
        assert config.models is not None
        assert config.system is not None

    def test_model_config(self, config):
        """Model configuration is correct."""
        assert config.models.embeddings.model == "qwen3-embedding:8b"
        assert config.models.llm.model == "qwen3:latest"
        assert config.models.vram.total == 12

    def test_system_config(self, config):
        """System configuration is correct."""
        assert config.system.chunking.chunk_size == 1024
        assert config.system.retrieval.max_results == 5


class TestPipelineInitialization:
    """Test pipeline initialization."""

    def test_pipeline_creates(self, pipeline):
        """Pipeline initializes successfully."""
        assert pipeline is not None

    def test_models_loaded(self, pipeline):
        """Models are loaded."""
        assert pipeline._embedder is not None
        assert pipeline._llm is not None

    def test_components_initialized(self, pipeline):
        """All components are initialized."""
        assert pipeline._parser is not None
        assert pipeline._chunker is not None
        assert pipeline._storage is not None
        assert pipeline._retriever is not None

    def test_get_stats(self, pipeline):
        """Pipeline stats are available."""
        stats = pipeline.get_stats()
        assert "models_loaded" in stats
        assert "config" in stats


class TestEmbedding:
    """Test embedding generation."""

    def test_embed_single_text(self, pipeline):
        """Can embed a single text."""
        text = "The Euler buckling formula predicts column stability."

        start = time.time()
        embedding = pipeline._embedder.embed([text])[0]
        elapsed_ms = (time.time() - start) * 1000

        assert embedding is not None
        assert len(embedding) == 256  # Matryoshka reduced
        assert elapsed_ms < 5000  # Should be fast

    def test_embed_batch(self, pipeline):
        """Can embed multiple texts."""
        texts = [
            "Stress is force per unit area.",
            "Strain is deformation per unit length.",
            "Young's modulus relates stress to strain.",
        ]

        start = time.time()
        embeddings = pipeline._embedder.embed(texts)
        elapsed_ms = (time.time() - start) * 1000

        assert len(embeddings) == len(texts)
        assert all(len(e) == 256 for e in embeddings)
        assert elapsed_ms < 10000  # Batch should be reasonably fast


class TestQuery:
    """Test query processing."""

    @pytest.mark.parametrize("query", TEST_QUERIES)
    def test_query_executes(self, pipeline, query):
        """Query executes without errors."""
        response = pipeline.query(query, max_results=5)

        assert response is not None
        assert response.answer is not None
        assert len(response.answer) > 0

    @pytest.mark.parametrize("query", TEST_QUERIES)
    def test_query_performance(self, pipeline, query):
        """Query meets performance targets."""
        response = pipeline.query(query, max_results=5)

        # Check latency targets from config
        assert response.total_time_ms < 10000  # <10s total (generous for tests)
        assert response.retrieval_time_ms < 2000  # <2s retrieval
        assert response.generation_time_ms < 15000  # <15s generation

    def test_query_returns_sources(self, pipeline):
        """Query returns source citations."""
        response = pipeline.query(
            "What is stress?",
            max_results=5,
            include_sources=True,
        )

        # Should have retrieved chunks
        assert response.sources_count > 0
        assert len(response.retrieved_chunks) > 0

    def test_query_tokens_tracked(self, pipeline):
        """Query tracks token usage."""
        response = pipeline.query("What is beam bending?")

        assert response.tokens_used > 0
        assert response.model_used == "qwen3:latest"


class TestQuality:
    """Test response quality."""

    def test_answer_relevance(self, pipeline):
        """Answers contain relevant keywords."""
        for query, keywords in EXPECTED_KEYWORDS.items():
            response = pipeline.query(query)

            # Answer should contain at least one relevant keyword
            answer_lower = response.answer.lower()
            found_keywords = [kw for kw in keywords if kw in answer_lower]

            assert len(found_keywords) > 0, (
                f"Answer for '{query}' should contain keywords: {keywords}\n"
                f"Got: {response.answer[:200]}"
            )

    def test_answer_not_empty(self, pipeline):
        """Answers are not empty."""
        response = pipeline.query("What is structural mechanics?")

        assert len(response.answer) > 50  # Substantial answer
        assert response.answer != ""
        assert response.answer.lower() != "i don't know"


class TestPerformanceBenchmark:
    """Performance benchmarking tests."""

    def test_cold_start_query(self, pipeline):
        """Measure cold start query performance."""
        query = "What is the difference between stress and strain?"

        start = time.time()
        response = pipeline.query(query)
        elapsed_ms = (time.time() - start) * 1000

        logger.info(f"Cold start query: {elapsed_ms:.0f}ms")
        logger.info(f"  Retrieval: {response.retrieval_time_ms:.0f}ms")
        logger.info(f"  Generation: {response.generation_time_ms:.0f}ms")

        # Log for analysis
        assert elapsed_ms < 20000  # First query can be slower

    def test_warm_query(self, pipeline):
        """Measure warm query performance."""
        # Warm up with first query
        pipeline.query("What is stress?")

        # Measure second query
        query = "What is strain?"

        start = time.time()
        response = pipeline.query(query)
        elapsed_ms = (time.time() - start) * 1000

        logger.info(f"Warm query: {elapsed_ms:.0f}ms")
        logger.info(f"  Retrieval: {response.retrieval_time_ms:.0f}ms")
        logger.info(f"  Generation: {response.generation_time_ms:.0f}ms")

        # Should be faster than cold start
        assert elapsed_ms < 15000

    def test_batch_queries(self, pipeline):
        """Test multiple sequential queries."""
        queries = TEST_QUERIES

        total_start = time.time()
        results = []

        for query in queries:
            response = pipeline.query(query)
            results.append(response)

        total_elapsed = (time.time() - total_start) * 1000

        avg_time = total_elapsed / len(queries)
        logger.info(f"Batch queries ({len(queries)}): {total_elapsed:.0f}ms total, {avg_time:.0f}ms avg")

        # All should complete
        assert len(results) == len(queries)
        assert all(r.answer for r in results)


@pytest.mark.skipif(
    not Path("data/raw").exists() or not list(Path("data/raw").glob("*.pdf")),
    reason="No test documents available in data/raw/",
)
class TestIndexing:
    """Test document indexing (requires test documents)."""

    def test_index_document(self, pipeline):
        """Can index a document."""
        # Find first PDF in data/raw
        test_docs = list(Path("data/raw").glob("*.pdf"))
        if not test_docs:
            pytest.skip("No test documents available")

        test_doc = test_docs[0]

        result = pipeline.index_document(
            str(test_doc),
            batch_size=32,
            show_progress=False,
        )

        assert result.success
        assert result.chunks_indexed > 0
        assert result.total_time_ms > 0

    def test_indexing_performance(self, pipeline):
        """Indexing meets performance targets."""
        test_docs = list(Path("data/raw").glob("*.pdf"))
        if not test_docs:
            pytest.skip("No test documents available")

        test_doc = test_docs[0]

        result = pipeline.index_document(str(test_doc))

        # Performance assertions
        assert result.parsing_time_ms > 0
        assert result.chunking_time_ms > 0
        assert result.embedding_time_ms > 0
        assert result.indexing_time_ms > 0

        # Log for analysis
        logger.info(f"Indexing performance for {test_doc.name}:")
        logger.info(f"  Parsing: {result.parsing_time_ms:.0f}ms")
        logger.info(f"  Chunking: {result.chunking_time_ms:.0f}ms")
        logger.info(f"  Embedding: {result.embedding_time_ms:.0f}ms")
        logger.info(f"  Indexing: {result.indexing_time_ms:.0f}ms")
        logger.info(f"  Total: {result.total_time_ms:.0f}ms")
        logger.info(f"  Chunks/sec: {result.chunks_indexed / (result.total_time_ms / 1000):.1f}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
