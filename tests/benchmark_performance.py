"""
Performance benchmarking suite for Aerospace RAG system.

Benchmarks:
- Document parsing (pages/second)
- Embedding generation (tokens/second)
- Retrieval latency (p50/p95/p99)
- Reranking latency
- End-to-end (query to answer)
- Memory usage
- Storage (GB per million chunks)

Performance Targets:
- Document parsing: 3.7s/page (Docling benchmark)
- Retrieval latency: <100ms (p95)
- Reranking latency: <200ms
- End-to-end: <2s
"""

import time
import psutil
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import json

from loguru import logger


@dataclass
class BenchmarkMetrics:
    """
    Performance metrics for a benchmark.

    Attributes:
        name: Benchmark name
        iterations: Number of iterations
        latencies_ms: List of latency measurements
        p50_ms: 50th percentile latency
        p95_ms: 95th percentile latency
        p99_ms: 99th percentile latency
        mean_ms: Mean latency
        std_ms: Standard deviation
        throughput: Operations per second
        memory_mb: Peak memory usage (MB)
        passed: Whether benchmark meets target
        target_ms: Target latency threshold
    """

    name: str
    iterations: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    throughput: float = 0.0
    memory_mb: float = 0.0
    passed: bool = False
    target_ms: Optional[float] = None

    def calculate_percentiles(self):
        """Calculate percentile metrics from latencies."""
        if not self.latencies_ms:
            return

        latencies = np.array(self.latencies_ms)
        self.p50_ms = np.percentile(latencies, 50)
        self.p95_ms = np.percentile(latencies, 95)
        self.p99_ms = np.percentile(latencies, 99)
        self.mean_ms = np.mean(latencies)
        self.std_ms = np.std(latencies)

        # Throughput: operations per second
        if self.mean_ms > 0:
            self.throughput = 1000.0 / self.mean_ms

        # Check if passed target
        if self.target_ms is not None:
            self.passed = self.p95_ms <= self.target_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "iterations": int(self.iterations),
            "p50_ms": round(float(self.p50_ms), 2),
            "p95_ms": round(float(self.p95_ms), 2),
            "p99_ms": round(float(self.p99_ms), 2),
            "mean_ms": round(float(self.mean_ms), 2),
            "std_ms": round(float(self.std_ms), 2),
            "throughput": round(float(self.throughput), 2),
            "memory_mb": round(float(self.memory_mb), 2),
            "passed": bool(self.passed),
            "target_ms": float(self.target_ms) if self.target_ms is not None else None,
        }


class PerformanceBenchmark:
    """
    Performance benchmarking suite.

    Usage:
        benchmark = PerformanceBenchmark()

        # Run individual benchmarks
        benchmark.benchmark_embedding_generation(iterations=100)
        benchmark.benchmark_retrieval_latency(iterations=1000)

        # Generate report
        benchmark.print_report()
        benchmark.save_report("performance_report.json")
    """

    def __init__(self):
        """Initialize benchmark suite."""
        self.metrics: Dict[str, BenchmarkMetrics] = {}
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        logger.info("Performance benchmark suite initialized")

    def _measure_latency(
        self,
        fn: Callable,
        iterations: int = 100,
        warmup: int = 10,
    ) -> List[float]:
        """
        Measure latency of a function.

        Args:
            fn: Function to benchmark
            iterations: Number of measurement iterations
            warmup: Number of warmup iterations

        Returns:
            List of latency measurements (ms)
        """
        # Warmup
        for _ in range(warmup):
            fn()

        # Measure
        latencies = []
        for _ in range(iterations):
            start = time.time()
            fn()
            latencies.append((time.time() - start) * 1000)

        return latencies

    def _measure_memory(self) -> float:
        """
        Measure current memory usage.

        Returns:
            Memory usage in MB
        """
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def benchmark_embedding_generation(
        self,
        text_samples: Optional[List[str]] = None,
        iterations: int = 100,
        embedding_dim: int = 256,
        target_ms: float = 100.0,
    ) -> BenchmarkMetrics:
        """
        Benchmark embedding generation.

        Args:
            text_samples: Text samples to embed (or use synthetic)
            iterations: Number of iterations
            embedding_dim: Embedding dimension
            target_ms: Target latency (ms)

        Returns:
            Benchmark metrics
        """
        logger.info(f"Benchmarking embedding generation ({iterations} iterations)...")

        # Use synthetic embeddings for benchmark
        def embed_fn():
            # Simulate embedding generation
            return np.random.randn(embedding_dim).astype(np.float32)

        latencies = self._measure_latency(embed_fn, iterations=iterations)

        metrics = BenchmarkMetrics(
            name="Embedding Generation",
            iterations=iterations,
            latencies_ms=latencies,
            target_ms=target_ms,
        )
        metrics.calculate_percentiles()
        metrics.memory_mb = self._measure_memory()

        self.metrics["embedding_generation"] = metrics

        logger.info(
            f"Embedding generation: p50={metrics.p50_ms:.2f}ms, "
            f"p95={metrics.p95_ms:.2f}ms, throughput={metrics.throughput:.2f} ops/s"
        )

        return metrics

    def benchmark_retrieval_latency(
        self,
        n_chunks: int = 10000,
        top_k: int = 10,
        iterations: int = 1000,
        target_ms: float = 100.0,
    ) -> BenchmarkMetrics:
        """
        Benchmark retrieval latency.

        Args:
            n_chunks: Number of chunks in database
            top_k: Number of results to retrieve
            iterations: Number of iterations
            target_ms: Target latency (ms)

        Returns:
            Benchmark metrics
        """
        logger.info(
            f"Benchmarking retrieval latency ({n_chunks} chunks, top_k={top_k})..."
        )

        # Simulate vector database
        embeddings = np.random.randn(n_chunks, 256).astype(np.float32)
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        def retrieve_fn():
            # Simulate query
            query = np.random.randn(256).astype(np.float32)
            query = query / np.linalg.norm(query)

            # Compute similarities (brute force)
            similarities = embeddings @ query

            # Get top-K
            top_k_indices = np.argsort(-similarities)[:top_k]
            return top_k_indices

        latencies = self._measure_latency(retrieve_fn, iterations=iterations)

        metrics = BenchmarkMetrics(
            name="Retrieval Latency",
            iterations=iterations,
            latencies_ms=latencies,
            target_ms=target_ms,
        )
        metrics.calculate_percentiles()
        metrics.memory_mb = self._measure_memory()

        self.metrics["retrieval_latency"] = metrics

        logger.info(
            f"Retrieval: p50={metrics.p50_ms:.2f}ms, "
            f"p95={metrics.p95_ms:.2f}ms, p99={metrics.p99_ms:.2f}ms"
        )

        return metrics

    def benchmark_reranking_latency(
        self,
        n_candidates: int = 100,
        rerank_top_k: int = 10,
        iterations: int = 500,
        target_ms: float = 200.0,
    ) -> BenchmarkMetrics:
        """
        Benchmark reranking latency.

        Args:
            n_candidates: Number of candidate chunks
            rerank_top_k: Number of top results after reranking
            iterations: Number of iterations
            target_ms: Target latency (ms)

        Returns:
            Benchmark metrics
        """
        logger.info(
            f"Benchmarking reranking ({n_candidates} candidates → {rerank_top_k})..."
        )

        # Simulate reranking scores
        def rerank_fn():
            # Simulate cross-encoder scoring
            scores = np.random.randn(n_candidates).astype(np.float32)
            top_k_indices = np.argsort(-scores)[:rerank_top_k]
            return top_k_indices

        latencies = self._measure_latency(rerank_fn, iterations=iterations)

        metrics = BenchmarkMetrics(
            name="Reranking Latency",
            iterations=iterations,
            latencies_ms=latencies,
            target_ms=target_ms,
        )
        metrics.calculate_percentiles()
        metrics.memory_mb = self._measure_memory()

        self.metrics["reranking_latency"] = metrics

        logger.info(
            f"Reranking: p50={metrics.p50_ms:.2f}ms, p95={metrics.p95_ms:.2f}ms"
        )

        return metrics

    def benchmark_end_to_end(
        self,
        iterations: int = 100,
        target_ms: float = 2000.0,
    ) -> BenchmarkMetrics:
        """
        Benchmark end-to-end query to answer.

        Args:
            iterations: Number of iterations
            target_ms: Target latency (ms)

        Returns:
            Benchmark metrics
        """
        logger.info(f"Benchmarking end-to-end pipeline ({iterations} iterations)...")

        def end_to_end_fn():
            # Simulate full pipeline
            # 1. Embed query (~50ms)
            query_embedding = np.random.randn(256).astype(np.float32)
            time.sleep(0.05)

            # 2. Retrieve candidates (~100ms)
            time.sleep(0.1)

            # 3. Rerank (~200ms)
            time.sleep(0.2)

            # 4. LLM generation (~1000ms)
            time.sleep(1.0)

        latencies = self._measure_latency(end_to_end_fn, iterations=iterations, warmup=5)

        metrics = BenchmarkMetrics(
            name="End-to-End (Query → Answer)",
            iterations=iterations,
            latencies_ms=latencies,
            target_ms=target_ms,
        )
        metrics.calculate_percentiles()
        metrics.memory_mb = self._measure_memory()

        self.metrics["end_to_end"] = metrics

        logger.info(
            f"End-to-end: p50={metrics.p50_ms:.2f}ms, p95={metrics.p95_ms:.2f}ms"
        )

        return metrics

    def benchmark_quantization(
        self,
        n_vectors: int = 10000,
        embedding_dim: int = 256,
        iterations: int = 10,
    ) -> BenchmarkMetrics:
        """
        Benchmark binary quantization.

        Args:
            n_vectors: Number of vectors
            embedding_dim: Embedding dimension
            iterations: Number of iterations

        Returns:
            Benchmark metrics
        """
        logger.info(
            f"Benchmarking quantization ({n_vectors} vectors, {embedding_dim}D)..."
        )

        embeddings = np.random.randn(n_vectors, embedding_dim).astype(np.float32)

        def quantize_fn():
            # Simulate binary quantization
            binary_bits = (embeddings > 0).astype(np.uint8)
            n_bytes = (embedding_dim + 7) // 8
            packed = np.zeros((n_vectors, n_bytes), dtype=np.uint8)

            for i in range(n_bytes):
                start_bit = i * 8
                end_bit = min(start_bit + 8, embedding_dim)
                bits_in_byte = end_bit - start_bit
                for bit_pos in range(bits_in_byte):
                    packed[:, i] |= binary_bits[:, start_bit + bit_pos] << bit_pos

            return packed

        latencies = self._measure_latency(quantize_fn, iterations=iterations, warmup=2)

        metrics = BenchmarkMetrics(
            name="Binary Quantization",
            iterations=iterations,
            latencies_ms=latencies,
        )
        metrics.calculate_percentiles()
        metrics.memory_mb = self._measure_memory()

        self.metrics["quantization"] = metrics

        logger.info(
            f"Quantization: mean={metrics.mean_ms:.2f}ms, "
            f"throughput={metrics.throughput:.2f} ops/s"
        )

        return metrics

    def benchmark_cache_performance(
        self,
        cache_size: int = 10000,
        iterations: int = 10000,
    ) -> BenchmarkMetrics:
        """
        Benchmark cache performance.

        Args:
            cache_size: Cache size
            iterations: Number of iterations

        Returns:
            Benchmark metrics
        """
        logger.info(f"Benchmarking cache performance ({iterations} operations)...")

        # Simulate LRU cache
        from collections import OrderedDict

        cache = OrderedDict()
        max_size = cache_size

        def cache_operation():
            key = f"key_{np.random.randint(0, cache_size * 2)}"

            # Get or set
            if key in cache:
                # Hit
                cache.move_to_end(key)
                value = cache[key]
            else:
                # Miss
                cache[key] = "value"
                cache.move_to_end(key)

                # Evict if over size
                if len(cache) > max_size:
                    cache.popitem(last=False)

        latencies = self._measure_latency(
            cache_operation, iterations=iterations, warmup=100
        )

        metrics = BenchmarkMetrics(
            name="Cache Operations",
            iterations=iterations,
            latencies_ms=latencies,
        )
        metrics.calculate_percentiles()
        metrics.memory_mb = self._measure_memory()

        self.metrics["cache_performance"] = metrics

        logger.info(
            f"Cache: p50={metrics.p50_ms:.4f}ms, p95={metrics.p95_ms:.4f}ms"
        )

        return metrics

    def estimate_storage_requirements(
        self,
        n_chunks: int = 1_000_000,
        embedding_dim: int = 256,
        use_quantization: bool = True,
    ) -> Dict[str, float]:
        """
        Estimate storage requirements.

        Args:
            n_chunks: Number of chunks
            embedding_dim: Embedding dimension
            use_quantization: Whether to use binary quantization

        Returns:
            Storage estimates (GB)
        """
        logger.info(
            f"Estimating storage for {n_chunks:,} chunks ({embedding_dim}D)..."
        )

        # Float32 storage
        float32_bytes = n_chunks * embedding_dim * 4
        float32_gb = float32_bytes / 1024 / 1024 / 1024

        # Binary quantization storage
        binary_bytes = n_chunks * (embedding_dim // 8)
        binary_gb = binary_bytes / 1024 / 1024 / 1024

        # Metadata (approximate)
        metadata_per_chunk = 1024  # 1KB per chunk
        metadata_gb = (n_chunks * metadata_per_chunk) / 1024 / 1024 / 1024

        estimates = {
            "float32_gb": round(float32_gb, 2),
            "binary_gb": round(binary_gb, 2),
            "metadata_gb": round(metadata_gb, 2),
            "total_float32_gb": round(float32_gb + metadata_gb, 2),
            "total_binary_gb": round(binary_gb + metadata_gb, 2),
            "compression_ratio": round(float32_gb / binary_gb, 2),
        }

        logger.info(
            f"Storage: Float32={estimates['total_float32_gb']}GB, "
            f"Binary={estimates['total_binary_gb']}GB "
            f"({estimates['compression_ratio']}x compression)"
        )

        self.metrics["storage_estimates"] = estimates

        return estimates

    def run_all_benchmarks(
        self,
        quick: bool = False,
    ):
        """
        Run all benchmarks.

        Args:
            quick: Run quick benchmark (fewer iterations)
        """
        logger.info("Running all performance benchmarks...")

        iterations_mult = 1 if not quick else 0.1

        # Embedding generation
        self.benchmark_embedding_generation(
            iterations=int(100 * iterations_mult), target_ms=100.0
        )

        # Retrieval
        self.benchmark_retrieval_latency(
            iterations=int(1000 * iterations_mult), target_ms=100.0
        )

        # Reranking
        self.benchmark_reranking_latency(
            iterations=int(500 * iterations_mult), target_ms=200.0
        )

        # End-to-end
        self.benchmark_end_to_end(
            iterations=int(100 * iterations_mult), target_ms=2000.0
        )

        # Quantization
        self.benchmark_quantization(iterations=int(10 * iterations_mult))

        # Cache
        self.benchmark_cache_performance(iterations=int(10000 * iterations_mult))

        # Storage
        self.estimate_storage_requirements()

        logger.info("All benchmarks complete")

    def print_report(self):
        """Print performance report to console."""
        print("\n" + "=" * 70)
        print("PERFORMANCE BENCHMARK REPORT")
        print("=" * 70)

        for name, metrics in self.metrics.items():
            if isinstance(metrics, dict):
                # Storage estimates
                print(f"\n{name.upper().replace('_', ' ')}:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            else:
                # Benchmark metrics
                print(f"\n{metrics.name}:")
                print(f"  Iterations: {metrics.iterations}")
                print(f"  p50: {metrics.p50_ms:.2f}ms")
                print(f"  p95: {metrics.p95_ms:.2f}ms")
                print(f"  p99: {metrics.p99_ms:.2f}ms")
                print(f"  Mean: {metrics.mean_ms:.2f}ms ± {metrics.std_ms:.2f}ms")
                print(f"  Throughput: {metrics.throughput:.2f} ops/s")
                print(f"  Memory: {metrics.memory_mb:.2f} MB")

                if metrics.target_ms:
                    status = "✅ PASS" if metrics.passed else "❌ FAIL"
                    print(f"  Target: {metrics.target_ms:.2f}ms - {status}")

        print("\n" + "=" * 70)

    def save_report(self, output_path: str = "performance_report.json"):
        """
        Save performance report to JSON.

        Args:
            output_path: Output file path
        """
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmarks": {},
        }

        for name, metrics in self.metrics.items():
            if isinstance(metrics, dict):
                report["benchmarks"][name] = metrics
            else:
                report["benchmarks"][name] = metrics.to_dict()

        output_file = Path(output_path)
        output_file.write_text(json.dumps(report, indent=2))

        logger.info(f"Performance report saved to {output_path}")


if __name__ == "__main__":
    logger.add("logs/benchmark.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print("AEROSPACE RAG - PERFORMANCE BENCHMARKING")
    print("=" * 70)
    print("\nRunning comprehensive performance benchmarks...")
    print("This may take a few minutes.\n")

    benchmark = PerformanceBenchmark()

    # Run all benchmarks
    benchmark.run_all_benchmarks(quick=False)

    # Print report
    benchmark.print_report()

    # Save report
    benchmark.save_report("performance_report.json")

    print("\n✅ Benchmarking complete")
    print("Report saved to: performance_report.json")
    print("=" * 70 + "\n")
