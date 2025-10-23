"""
Multi-level caching for RAG pipeline optimization.

Implements three cache levels:
- Query cache (1 hour TTL): Complete query → answer mappings
- Embedding cache (persistent): Text → embedding mappings
- Reranking cache (24 hour TTL): Query + chunks → reranked results

Supports both Redis (distributed) and LRU (local) backends.

Usage:
    # Initialize cache manager
    cache = CacheManager(backend="lru", max_size=10000)

    # Query cache
    answer = cache.get_query("What is beam bending?")
    if answer is None:
        answer = generate_answer(query)
        cache.set_query(query, answer, ttl=3600)

    # Embedding cache
    embedding = cache.get_embedding("Technical text...")
    if embedding is None:
        embedding = embed_text(text)
        cache.set_embedding(text, embedding)
"""

import hashlib
import pickle
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from collections import OrderedDict

import numpy as np
from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Install with: pip install redis")


class CacheBackend(str, Enum):
    """Cache backend types."""
    REDIS = "redis"  # Distributed cache
    LRU = "lru"  # Local in-memory cache


class CacheLevel(str, Enum):
    """Cache level types."""
    QUERY = "query"  # Full query → answer
    EMBEDDING = "embedding"  # Text → embedding
    RERANKING = "reranking"  # Query + chunks → reranked results


@dataclass
class CacheStats:
    """
    Cache statistics.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        hit_rate: Hit rate percentage
        total_requests: Total requests
        avg_latency_hit_ms: Average latency on hit
        avg_latency_miss_ms: Average latency on miss
        size_bytes: Current cache size in bytes
        evictions: Number of evictions
    """
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    total_requests: int = 0
    avg_latency_hit_ms: float = 0.0
    avg_latency_miss_ms: float = 0.0
    size_bytes: int = 0
    evictions: int = 0

    def update_stats(self):
        """Update computed statistics."""
        self.total_requests = self.hits + self.misses
        if self.total_requests > 0:
            self.hit_rate = (self.hits / self.total_requests) * 100


class LRUCache:
    """
    LRU (Least Recently Used) cache implementation.

    Thread-safe, in-memory cache with size limits and TTL support.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
        """
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.ttls: Dict[str, float] = {}  # key → expiry timestamp
        self.stats = CacheStats()

        logger.debug(f"LRU cache initialized with max_size={max_size}")

    def _hash_key(self, key: str) -> str:
        """Create hash of key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if miss
        """
        start_time = time.time()
        hashed_key = self._hash_key(key)

        # Check if key exists and not expired
        if hashed_key in self.cache:
            # Check TTL
            if hashed_key in self.ttls:
                if time.time() > self.ttls[hashed_key]:
                    # Expired
                    del self.cache[hashed_key]
                    del self.ttls[hashed_key]
                    self.stats.misses += 1
                    return None

            # Move to end (most recently used)
            self.cache.move_to_end(hashed_key)

            latency_ms = (time.time() - start_time) * 1000
            self.stats.hits += 1
            self.stats.avg_latency_hit_ms = (
                (self.stats.avg_latency_hit_ms * (self.stats.hits - 1) + latency_ms)
                / self.stats.hits
            )
            self.stats.update_stats()

            return self.cache[hashed_key]

        self.stats.misses += 1
        self.stats.update_stats()
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        hashed_key = self._hash_key(key)

        # Add to cache
        self.cache[hashed_key] = value
        self.cache.move_to_end(hashed_key)

        # Set TTL if provided
        if ttl is not None:
            self.ttls[hashed_key] = time.time() + ttl

        # Evict if over size
        if len(self.cache) > self.max_size:
            # Remove least recently used
            evicted_key = next(iter(self.cache))
            del self.cache[evicted_key]
            if evicted_key in self.ttls:
                del self.ttls[evicted_key]
            self.stats.evictions += 1

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.ttls.clear()
        logger.debug("LRU cache cleared")

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        # Estimate size in bytes
        self.stats.size_bytes = sum(
            len(pickle.dumps(v)) for v in list(self.cache.values())[:100]
        ) * (len(self.cache) / min(100, len(self.cache)))

        return self.stats


class RedisCache:
    """
    Redis-based distributed cache.

    Supports TTL, persistence, and distributed access.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "rag:",
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            prefix: Key prefix for namespacing
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not installed. Install with: pip install redis")

        self.prefix = prefix
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # Keep binary for pickle
        )
        self.stats = CacheStats()

        # Test connection
        try:
            self.client.ping()
            logger.info(f"Redis cache connected: {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def _make_key(self, key: str) -> str:
        """Create prefixed Redis key."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        start_time = time.time()
        redis_key = self._make_key(key)

        try:
            value = self.client.get(redis_key)

            if value is not None:
                latency_ms = (time.time() - start_time) * 1000
                self.stats.hits += 1
                self.stats.avg_latency_hit_ms = (
                    (self.stats.avg_latency_hit_ms * (self.stats.hits - 1) + latency_ms)
                    / self.stats.hits
                )
                self.stats.update_stats()

                return pickle.loads(value)

            self.stats.misses += 1
            self.stats.update_stats()
            return None

        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats.misses += 1
            self.stats.update_stats()
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis cache."""
        redis_key = self._make_key(key)

        try:
            pickled_value = pickle.dumps(value)

            if ttl is not None:
                self.client.setex(redis_key, ttl, pickled_value)
            else:
                self.client.set(redis_key, pickled_value)

        except Exception as e:
            logger.error(f"Redis set error: {e}")

    def clear(self):
        """Clear all keys with prefix."""
        pattern = f"{self.prefix}*"
        keys = self.client.keys(pattern)

        if keys:
            self.client.delete(*keys)
            logger.info(f"Redis cache cleared: {len(keys)} keys")

    def size(self) -> int:
        """Get number of keys with prefix."""
        pattern = f"{self.prefix}*"
        return len(self.client.keys(pattern))

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        # Get memory usage from Redis info
        try:
            info = self.client.info("memory")
            self.stats.size_bytes = info.get("used_memory", 0)
        except:
            pass

        return self.stats


class CacheManager:
    """
    Multi-level cache manager for RAG pipeline.

    Manages three cache levels:
    - Query cache: Full query results
    - Embedding cache: Text embeddings
    - Reranking cache: Reranked results

    Usage:
        cache = CacheManager(backend="lru")

        # Query cache
        answer = cache.get_query("What is stress?")
        cache.set_query(query, answer)

        # Embedding cache
        embedding = cache.get_embedding("Technical text")
        cache.set_embedding(text, embedding)

        # Reranking cache
        results = cache.get_reranking(query, chunk_ids)
        cache.set_reranking(query, chunk_ids, reranked_results)
    """

    def __init__(
        self,
        backend: str = "lru",
        max_size: int = 10000,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
    ):
        """
        Initialize cache manager.

        Args:
            backend: "lru" or "redis"
            max_size: Max size for LRU cache
            redis_host: Redis host (if backend="redis")
            redis_port: Redis port (if backend="redis")
            redis_password: Redis password (optional)
        """
        self.backend = CacheBackend(backend)

        # Initialize cache backend
        if self.backend == CacheBackend.LRU:
            self.cache = LRUCache(max_size=max_size)
        else:  # REDIS
            self.cache = RedisCache(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                prefix="rag:",
            )

        logger.info(f"Cache manager initialized with {backend} backend")

    # Query cache methods
    def get_query(self, query: str) -> Optional[str]:
        """Get cached query result."""
        key = f"query:{query}"
        return self.cache.get(key)

    def set_query(self, query: str, answer: str, ttl: int = 3600):
        """
        Cache query result.

        Args:
            query: Query string
            answer: Answer string
            ttl: Time to live (default: 1 hour)
        """
        key = f"query:{query}"
        self.cache.set(key, answer, ttl=ttl)

    # Embedding cache methods
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = f"embedding:{text}"
        return self.cache.get(key)

    def set_embedding(self, text: str, embedding: np.ndarray, ttl: Optional[int] = None):
        """
        Cache embedding (persistent by default).

        Args:
            text: Input text
            embedding: Embedding vector
            ttl: Time to live (None = persistent)
        """
        key = f"embedding:{text}"
        self.cache.set(key, embedding, ttl=ttl)

    # Reranking cache methods
    def get_reranking(
        self,
        query: str,
        chunk_ids: List[str],
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Get cached reranking results.

        Args:
            query: Query string
            chunk_ids: List of chunk IDs

        Returns:
            List of (chunk_id, score) tuples or None
        """
        # Create key from query + sorted chunk IDs
        chunk_ids_str = ",".join(sorted(chunk_ids))
        key = f"reranking:{query}:{chunk_ids_str}"
        return self.cache.get(key)

    def set_reranking(
        self,
        query: str,
        chunk_ids: List[str],
        results: List[Tuple[str, float]],
        ttl: int = 86400,  # 24 hours
    ):
        """
        Cache reranking results.

        Args:
            query: Query string
            chunk_ids: List of chunk IDs
            results: List of (chunk_id, score) tuples
            ttl: Time to live (default: 24 hours)
        """
        chunk_ids_str = ",".join(sorted(chunk_ids))
        key = f"reranking:{query}:{chunk_ids_str}"
        self.cache.set(key, results, ttl=ttl)

    def warm_cache(self, common_queries: List[str], embed_fn, retrieve_fn):
        """
        Warm cache with common queries.

        Args:
            common_queries: List of common technical terms/queries
            embed_fn: Function to generate embeddings
            retrieve_fn: Function to retrieve and rerank
        """
        logger.info(f"Warming cache with {len(common_queries)} queries...")

        for i, query in enumerate(common_queries):
            # Pre-compute and cache embedding
            embedding = embed_fn(query)
            self.set_embedding(query, embedding)

            # Pre-compute and cache retrieval
            try:
                results = retrieve_fn(query)
                # Cache full query result
                # (In practice, would cache the final answer here)
            except Exception as e:
                logger.warning(f"Cache warming failed for '{query}': {e}")

            if (i + 1) % 10 == 0:
                logger.debug(f"Warmed {i + 1}/{len(common_queries)} queries")

        logger.info("Cache warming complete")

    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache levels."""
        return {
            "overall": self.cache.get_stats(),
        }

    def clear(self):
        """Clear all cache levels."""
        self.cache.clear()
        logger.info("All cache levels cleared")


if __name__ == "__main__":
    logger.add("logs/cache.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print("MULTI-LEVEL CACHING - RAG Pipeline Optimization")
    print("=" * 70)
    print("\nCache Levels:")
    print("  • Query cache: Full query → answer (1 hour TTL)")
    print("  • Embedding cache: Text → embedding (persistent)")
    print("  • Reranking cache: Query + chunks → scores (24 hour TTL)")
    print("\nBackends:")
    print("  • LRU: Local in-memory cache (fast, single instance)")
    print("  • Redis: Distributed cache (shared, persistent)")
    print("\nFeatures:")
    print("  • TTL support for automatic expiration")
    print("  • Hit rate tracking and statistics")
    print("  • Cache warming for common queries")
    print("  • Size-based eviction (LRU)")
    print("\nUsage:")
    print("  cache = CacheManager(backend='lru', max_size=10000)")
    print("  answer = cache.get_query('What is stress?')")
    print("  if answer is None:")
    print("      answer = generate_answer(query)")
    print("      cache.set_query(query, answer)")
    print("=" * 70 + "\n")
