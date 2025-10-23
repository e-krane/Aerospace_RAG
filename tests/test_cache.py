"""
Tests for multi-level caching system.

Tests:
- LRU cache get/set operations
- TTL expiration behavior
- Size-based eviction (LRU policy)
- Cache statistics tracking
- Multi-level cache manager (query, embedding, reranking)
- Cache warming functionality
- Hit rate and latency measurements
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from src.utils.cache import (
    LRUCache,
    CacheManager,
    CacheBackend,
    CacheLevel,
    CacheStats,
)


class TestLRUCache:
    """Test LRU cache functionality."""

    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        cache = LRUCache(max_size=100)

        assert cache.max_size == 100
        assert cache.size() == 0
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    def test_cache_set_get(self):
        """Test basic set/get operations."""
        cache = LRUCache(max_size=100)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.size() == 2

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = LRUCache(max_size=100)

        result = cache.get("nonexistent")

        assert result is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

    def test_cache_hit_statistics(self):
        """Test cache hit statistics."""
        cache = LRUCache(max_size=100)

        cache.set("key1", "value1")

        # Hit
        result = cache.get("key1")
        assert result == "value1"
        assert cache.stats.hits == 1

        # Another hit
        result = cache.get("key1")
        assert cache.stats.hits == 2

        # Miss
        cache.get("nonexistent")
        assert cache.stats.misses == 1

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        cache = LRUCache(max_size=100)

        cache.set("key1", "value1")

        # 2 hits, 1 miss
        cache.get("key1")
        cache.get("key1")
        cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.total_requests == 3
        assert abs(stats.hit_rate - 66.67) < 0.1

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = LRUCache(max_size=100)

        # Set with 1 second TTL
        cache.set("key1", "value1", ttl=1)

        # Should exist immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("key1") is None
        assert cache.stats.misses == 1

    def test_size_based_eviction(self):
        """Test LRU eviction when max size exceeded."""
        cache = LRUCache(max_size=3)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.size() == 3

        # Add fourth item - should evict key1 (least recently used)
        cache.set("key4", "value4")

        assert cache.size() == 3
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_lru_ordering(self):
        """Test LRU ordering with access patterns."""
        cache = LRUCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4 - should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Still exists
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = LRUCache(max_size=100)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.size() == 2

        cache.clear()

        assert cache.size() == 0
        assert cache.get("key1") is None

    def test_cache_with_numpy_arrays(self):
        """Test caching numpy arrays."""
        cache = LRUCache(max_size=100)

        embedding = np.random.randn(256).astype(np.float32)

        cache.set("embedding1", embedding)

        result = cache.get("embedding1")

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, embedding)

    def test_eviction_count(self):
        """Test eviction tracking."""
        cache = LRUCache(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Evicts key1
        cache.set("key4", "value4")  # Evicts key2

        stats = cache.get_stats()
        assert stats.evictions == 2


class TestCacheManager:
    """Test multi-level cache manager."""

    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        manager = CacheManager(backend="lru", max_size=1000)

        assert manager.backend == CacheBackend.LRU
        assert isinstance(manager.cache, LRUCache)

    def test_query_cache(self):
        """Test query cache operations."""
        manager = CacheManager(backend="lru")

        query = "What is beam bending?"
        answer = "Beam bending is..."

        # Set query result
        manager.set_query(query, answer)

        # Get query result
        result = manager.get_query(query)

        assert result == answer

    def test_query_cache_miss(self):
        """Test query cache miss."""
        manager = CacheManager(backend="lru")

        result = manager.get_query("nonexistent query")

        assert result is None

    def test_embedding_cache(self):
        """Test embedding cache operations."""
        manager = CacheManager(backend="lru")

        text = "Technical text about aerospace"
        embedding = np.random.randn(256).astype(np.float32)

        # Set embedding
        manager.set_embedding(text, embedding)

        # Get embedding
        result = manager.get_embedding(text)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, embedding)

    def test_embedding_cache_persistent(self):
        """Test embedding cache is persistent (no TTL by default)."""
        manager = CacheManager(backend="lru")

        text = "Technical text"
        embedding = np.random.randn(256).astype(np.float32)

        # Set without TTL
        manager.set_embedding(text, embedding)

        # Should persist (we can't wait forever, but check it exists)
        result = manager.get_embedding(text)
        assert result is not None

    def test_reranking_cache(self):
        """Test reranking cache operations."""
        manager = CacheManager(backend="lru")

        query = "What is stress?"
        chunk_ids = ["chunk1", "chunk2", "chunk3"]
        results = [("chunk1", 0.95), ("chunk2", 0.87), ("chunk3", 0.72)]

        # Set reranking results
        manager.set_reranking(query, chunk_ids, results)

        # Get reranking results
        cached_results = manager.get_reranking(query, chunk_ids)

        assert cached_results == results

    def test_reranking_cache_chunk_order_invariant(self):
        """Test reranking cache is invariant to chunk order."""
        manager = CacheManager(backend="lru")

        query = "What is stress?"
        chunk_ids_1 = ["chunk1", "chunk2", "chunk3"]
        chunk_ids_2 = ["chunk3", "chunk1", "chunk2"]  # Different order
        results = [("chunk1", 0.95), ("chunk2", 0.87), ("chunk3", 0.72)]

        # Set with first order
        manager.set_reranking(query, chunk_ids_1, results)

        # Get with different order - should still hit cache
        cached_results = manager.get_reranking(query, chunk_ids_2)

        assert cached_results == results

    def test_query_cache_ttl(self):
        """Test query cache TTL (1 hour default)."""
        manager = CacheManager(backend="lru")

        query = "What is beam bending?"
        answer = "Beam bending is..."

        # Set with short TTL for testing
        manager.set_query(query, answer, ttl=1)

        # Should exist immediately
        assert manager.get_query(query) == answer

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert manager.get_query(query) is None

    def test_multiple_cache_levels(self):
        """Test all three cache levels work together."""
        manager = CacheManager(backend="lru")

        # Query cache
        manager.set_query("query1", "answer1")

        # Embedding cache
        embedding = np.random.randn(256).astype(np.float32)
        manager.set_embedding("text1", embedding)

        # Reranking cache
        manager.set_reranking("query2", ["c1", "c2"], [("c1", 0.9), ("c2", 0.8)])

        # All should be retrievable
        assert manager.get_query("query1") == "answer1"
        assert np.array_equal(manager.get_embedding("text1"), embedding)
        assert manager.get_reranking("query2", ["c1", "c2"]) == [("c1", 0.9), ("c2", 0.8)]

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        manager = CacheManager(backend="lru")

        # Generate some cache activity
        manager.set_query("query1", "answer1")
        manager.get_query("query1")  # Hit
        manager.get_query("query2")  # Miss

        stats = manager.get_stats()

        assert "overall" in stats
        assert stats["overall"].hits == 1
        assert stats["overall"].misses == 1

    def test_cache_clear(self):
        """Test clearing all cache levels."""
        manager = CacheManager(backend="lru")

        # Populate all cache levels
        manager.set_query("query1", "answer1")
        manager.set_embedding("text1", np.random.randn(256).astype(np.float32))
        manager.set_reranking("query2", ["c1"], [("c1", 0.9)])

        # Clear all
        manager.clear()

        # All should be gone
        assert manager.get_query("query1") is None
        assert manager.get_embedding("text1") is None
        assert manager.get_reranking("query2", ["c1"]) is None

    def test_warm_cache(self):
        """Test cache warming with common queries."""
        manager = CacheManager(backend="lru")

        # Mock embedding and retrieval functions
        def mock_embed_fn(text):
            return np.random.randn(256).astype(np.float32)

        def mock_retrieve_fn(query):
            return [("chunk1", 0.9), ("chunk2", 0.8)]

        common_queries = ["beam bending", "stress analysis", "moment of inertia"]

        # Warm cache
        manager.warm_cache(common_queries, mock_embed_fn, mock_retrieve_fn)

        # Check embeddings were cached
        for query in common_queries:
            cached_embedding = manager.get_embedding(query)
            assert cached_embedding is not None
            assert isinstance(cached_embedding, np.ndarray)


class TestCacheStats:
    """Test cache statistics dataclass."""

    def test_stats_initialization(self):
        """Test stats initialize correctly."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        assert stats.total_requests == 0

    def test_stats_update(self):
        """Test stats update computation."""
        stats = CacheStats()

        stats.hits = 7
        stats.misses = 3
        stats.update_stats()

        assert stats.total_requests == 10
        assert stats.hit_rate == 70.0


class TestCacheKeyHashing:
    """Test cache key hashing."""

    def test_same_key_same_hash(self):
        """Test same key produces same hash."""
        cache = LRUCache(max_size=100)

        cache.set("mykey", "value1")
        result1 = cache.get("mykey")

        cache.set("mykey", "value2")  # Overwrite
        result2 = cache.get("mykey")

        assert result2 == "value2"  # Updated value

    def test_different_keys_different_values(self):
        """Test different keys store different values."""
        cache = LRUCache(max_size=100)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"


class TestCachePerformance:
    """Test cache performance characteristics."""

    def test_large_cache_operations(self):
        """Test cache with large number of items."""
        cache = LRUCache(max_size=10000)

        # Add 5000 items
        for i in range(5000):
            cache.set(f"key{i}", f"value{i}")

        assert cache.size() == 5000

        # Access first 100 items
        for i in range(100):
            assert cache.get(f"key{i}") == f"value{i}"

    def test_cache_latency_tracking(self):
        """Test cache latency measurements."""
        cache = LRUCache(max_size=100)

        cache.set("key1", "value1")

        # Generate hits
        for _ in range(10):
            cache.get("key1")

        stats = cache.get_stats()

        # Should have measured hit latency
        assert stats.avg_latency_hit_ms >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
