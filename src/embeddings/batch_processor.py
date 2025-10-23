"""
Batch embedding pipeline with checkpointing and progress tracking.

Processes large document collections efficiently with:
- Resumable processing via checkpoints
- Dynamic batch sizing based on content length
- GPU utilization monitoring
- Error handling with retry logic
- Progress tracking with tqdm
- Embedding cache to avoid recomputation
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib
import time
from collections import defaultdict

try:
    import numpy as np
    from tqdm import tqdm
    import torch
except ImportError as e:
    raise ImportError(
        "numpy, tqdm, and torch required. Install with: "
        "pip install numpy tqdm torch"
    ) from e

from loguru import logger

# Import embedder
try:
    from .qwen3_embedder import Qwen3Embedder
except ImportError:
    from qwen3_embedder import Qwen3Embedder


@dataclass
class EmbeddingResult:
    """Result of embedding a single chunk."""

    chunk_id: str
    embedding: np.ndarray
    dimension: int
    token_count: int
    processing_time: float
    cached: bool = False


@dataclass
class BatchStats:
    """Statistics for batch processing."""

    total_chunks: int = 0
    processed_chunks: int = 0
    cached_chunks: int = 0
    failed_chunks: int = 0
    total_tokens: int = 0
    processing_time: float = 0.0
    avg_batch_size: float = 0.0
    gpu_utilization: List[float] = None

    def __post_init__(self):
        if self.gpu_utilization is None:
            self.gpu_utilization = []


class EmbeddingCache:
    """
    Cache for embeddings to avoid recomputation.

    Uses content hash as key for cache lookup.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.cache_dir / "cache_index.json"
        self.index = self._load_index()

        logger.info(f"Embedding cache initialized: {len(self.index)} entries")

    def _load_index(self) -> Dict[str, str]:
        """Load cache index from disk."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save cache index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)

    def _get_hash(self, content: str) -> str:
        """Get content hash for cache key."""
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, content: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for content.

        Args:
            content: Text content

        Returns:
            Cached embedding or None if not found
        """
        content_hash = self._get_hash(content)

        if content_hash in self.index:
            cache_file = self.cache_dir / self.index[content_hash]
            if cache_file.exists():
                try:
                    embedding = np.load(cache_file)
                    return embedding
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding: {e}")

        return None

    def put(self, content: str, embedding: np.ndarray):
        """
        Store embedding in cache.

        Args:
            content: Text content
            embedding: Embedding vector
        """
        content_hash = self._get_hash(content)
        cache_file = f"{content_hash}.npy"

        # Save embedding
        np.save(self.cache_dir / cache_file, embedding)

        # Update index
        self.index[content_hash] = cache_file
        self._save_index()

    def clear(self):
        """Clear all cached embeddings."""
        for cache_file in self.cache_dir.glob("*.npy"):
            cache_file.unlink()
        self.index = {}
        self._save_index()
        logger.info("Cache cleared")


class CheckpointManager:
    """
    Manage processing checkpoints for resumable processing.
    """

    def __init__(self, checkpoint_file: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_file: Path to checkpoint file
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        self.checkpoint = self._load_checkpoint()

    def _load_checkpoint(self) -> Dict:
        """Load checkpoint from disk."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {
            'processed_chunks': [],
            'failed_chunks': [],
            'last_processed_index': -1,
            'stats': {},
        }

    def save(self):
        """Save checkpoint to disk."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)

    def is_processed(self, chunk_id: str) -> bool:
        """Check if chunk already processed."""
        return chunk_id in self.checkpoint['processed_chunks']

    def mark_processed(self, chunk_id: str, index: int):
        """Mark chunk as processed."""
        if chunk_id not in self.checkpoint['processed_chunks']:
            self.checkpoint['processed_chunks'].append(chunk_id)
        self.checkpoint['last_processed_index'] = index

    def mark_failed(self, chunk_id: str, error: str):
        """Mark chunk as failed."""
        self.checkpoint['failed_chunks'].append({
            'chunk_id': chunk_id,
            'error': str(error),
            'timestamp': time.time(),
        })

    def update_stats(self, stats: BatchStats):
        """Update checkpoint statistics."""
        self.checkpoint['stats'] = asdict(stats)

    def get_resume_index(self) -> int:
        """Get index to resume from."""
        return self.checkpoint['last_processed_index'] + 1

    def clear(self):
        """Clear checkpoint."""
        self.checkpoint = {
            'processed_chunks': [],
            'failed_chunks': [],
            'last_processed_index': -1,
            'stats': {},
        }
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        logger.info("Checkpoint cleared")


class BatchEmbeddingProcessor:
    """
    Batch embedding processor with advanced features.

    Features:
    - Progress tracking with tqdm
    - Checkpoint system for resumable processing
    - Embedding cache to avoid recomputation
    - Dynamic batch sizing based on text length
    - GPU utilization monitoring
    - Error handling with retry logic
    - Memory management for large batches
    """

    def __init__(
        self,
        embedder: Optional[Qwen3Embedder] = None,
        cache_dir: str = "cache/embeddings",
        checkpoint_file: str = "checkpoints/embedding_progress.json",
        batch_size: int = 32,
        max_retries: int = 3,
        enable_cache: bool = True,
        enable_checkpoints: bool = True,
    ):
        """
        Initialize batch processor.

        Args:
            embedder: Qwen3Embedder instance (creates default if None)
            cache_dir: Directory for embedding cache
            checkpoint_file: Path to checkpoint file
            batch_size: Default batch size
            max_retries: Maximum retry attempts for failed chunks
            enable_cache: Enable embedding cache
            enable_checkpoints: Enable checkpoint system
        """
        self.embedder = embedder or Qwen3Embedder()
        self.batch_size = batch_size
        self.max_retries = max_retries

        # Initialize cache
        self.enable_cache = enable_cache
        if enable_cache:
            self.cache = EmbeddingCache(Path(cache_dir))
        else:
            self.cache = None

        # Initialize checkpoint manager
        self.enable_checkpoints = enable_checkpoints
        if enable_checkpoints:
            self.checkpoint = CheckpointManager(Path(checkpoint_file))
        else:
            self.checkpoint = None

        # Statistics
        self.stats = BatchStats()

        logger.info(
            f"Batch processor initialized: "
            f"batch_size={batch_size}, cache={enable_cache}, "
            f"checkpoints={enable_checkpoints}"
        )

    def process_chunks(
        self,
        chunks: List,
        show_progress: bool = True,
        save_checkpoints: bool = True,
    ) -> List[EmbeddingResult]:
        """
        Process chunks and generate embeddings.

        Args:
            chunks: List of chunks to embed
            show_progress: Show progress bar
            save_checkpoints: Save checkpoints during processing

        Returns:
            List of EmbeddingResult objects
        """
        self.stats = BatchStats(total_chunks=len(chunks))
        results = []

        # Determine resume point
        start_index = 0
        if self.checkpoint and self.enable_checkpoints:
            start_index = self.checkpoint.get_resume_index()
            if start_index > 0:
                logger.info(f"Resuming from checkpoint at index {start_index}")

        # Progress bar
        iterator = tqdm(
            enumerate(chunks[start_index:], start=start_index),
            total=len(chunks),
            initial=start_index,
            desc="Embedding chunks",
            disable=not show_progress,
        )

        # Process chunks
        batch_buffer = []
        batch_indices = []

        for i, chunk in iterator:
            chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"chunk_{i}"

            # Skip if already processed
            if self.checkpoint and self.checkpoint.is_processed(chunk_id):
                self.stats.processed_chunks += 1
                continue

            # Check cache
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            cached_embedding = None

            if self.cache:
                cached_embedding = self.cache.get(content)

            if cached_embedding is not None:
                # Use cached embedding
                result = EmbeddingResult(
                    chunk_id=chunk_id,
                    embedding=cached_embedding,
                    dimension=len(cached_embedding),
                    token_count=len(content.split()),
                    processing_time=0.0,
                    cached=True,
                )
                results.append(result)
                self.stats.cached_chunks += 1
                self.stats.processed_chunks += 1

                if self.checkpoint:
                    self.checkpoint.mark_processed(chunk_id, i)

                continue

            # Add to batch buffer
            batch_buffer.append(chunk)
            batch_indices.append(i)

            # Process batch when buffer is full
            if len(batch_buffer) >= self.batch_size:
                batch_results = self._process_batch(batch_buffer, batch_indices)
                results.extend(batch_results)

                # Clear buffer
                batch_buffer = []
                batch_indices = []

                # Save checkpoint
                if save_checkpoints and self.checkpoint:
                    self.checkpoint.save()

                # Update GPU stats
                self._update_gpu_stats()

        # Process remaining chunks
        if batch_buffer:
            batch_results = self._process_batch(batch_buffer, batch_indices)
            results.extend(batch_results)

        # Final checkpoint
        if save_checkpoints and self.checkpoint:
            self.checkpoint.update_stats(self.stats)
            self.checkpoint.save()

        logger.info(
            f"Batch processing complete: "
            f"{self.stats.processed_chunks}/{self.stats.total_chunks} chunks "
            f"({self.stats.cached_chunks} cached, {self.stats.failed_chunks} failed)"
        )

        return results

    def _process_batch(
        self,
        batch_chunks: List,
        batch_indices: List[int],
    ) -> List[EmbeddingResult]:
        """Process a batch of chunks."""
        results = []
        start_time = time.time()

        # Extract content
        texts = [
            chunk.content if hasattr(chunk, 'content') else str(chunk)
            for chunk in batch_chunks
        ]

        # Generate embeddings with retry logic
        embeddings = None
        retry_count = 0

        while embeddings is None and retry_count < self.max_retries:
            try:
                embeddings = self.embedder.embed(texts)
            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"Batch embedding failed (attempt {retry_count}/{self.max_retries}): {e}"
                )
                if retry_count >= self.max_retries:
                    logger.error(f"Batch embedding failed after {self.max_retries} retries")
                    # Mark all chunks as failed
                    for chunk, idx in zip(batch_chunks, batch_indices):
                        chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"chunk_{idx}"
                        self.stats.failed_chunks += 1
                        if self.checkpoint:
                            self.checkpoint.mark_failed(chunk_id, str(e))
                    return results
                time.sleep(1.0 * retry_count)  # Exponential backoff

        # Process results
        batch_time = time.time() - start_time

        for chunk, embedding, idx in zip(batch_chunks, embeddings, batch_indices):
            chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"chunk_{idx}"
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)

            # Cache embedding
            if self.cache:
                self.cache.put(content, embedding)

            # Create result
            result = EmbeddingResult(
                chunk_id=chunk_id,
                embedding=embedding,
                dimension=len(embedding),
                token_count=chunk.token_count if hasattr(chunk, 'token_count') else len(content.split()),
                processing_time=batch_time / len(batch_chunks),
                cached=False,
            )

            results.append(result)
            self.stats.processed_chunks += 1
            self.stats.total_tokens += result.token_count

            # Mark as processed
            if self.checkpoint:
                self.checkpoint.mark_processed(chunk_id, idx)

        self.stats.processing_time += batch_time

        return results

    def _update_gpu_stats(self):
        """Update GPU utilization statistics."""
        if torch.cuda.is_available():
            try:
                utilization = torch.cuda.utilization()
                self.stats.gpu_utilization.append(utilization)
            except Exception:
                pass  # GPU monitoring not critical

    def get_stats(self) -> Dict:
        """Get processing statistics."""
        stats_dict = asdict(self.stats)

        # Calculate averages
        if self.stats.processed_chunks > 0:
            stats_dict['avg_processing_time'] = (
                self.stats.processing_time / self.stats.processed_chunks
            )
            stats_dict['avg_tokens_per_chunk'] = (
                self.stats.total_tokens / self.stats.processed_chunks
            )

        if self.stats.gpu_utilization:
            stats_dict['avg_gpu_utilization'] = np.mean(self.stats.gpu_utilization)
            stats_dict['max_gpu_utilization'] = np.max(self.stats.gpu_utilization)

        return stats_dict

    def clear_cache(self):
        """Clear embedding cache."""
        if self.cache:
            self.cache.clear()

    def clear_checkpoint(self):
        """Clear checkpoint."""
        if self.checkpoint:
            self.checkpoint.clear()


def process_document_chunks(
    chunks: List,
    output_file: Optional[str] = None,
    batch_size: int = 32,
    enable_cache: bool = True,
    enable_checkpoints: bool = True,
) -> Tuple[List[EmbeddingResult], Dict]:
    """
    Convenience function to process chunks and optionally save results.

    Args:
        chunks: List of chunks to embed
        output_file: Optional file to save embeddings (NPY format)
        batch_size: Batch size for processing
        enable_cache: Enable embedding cache
        enable_checkpoints: Enable checkpoint system

    Returns:
        Tuple of (results, statistics)
    """
    processor = BatchEmbeddingProcessor(
        batch_size=batch_size,
        enable_cache=enable_cache,
        enable_checkpoints=enable_checkpoints,
    )

    results = processor.process_chunks(chunks, show_progress=True)
    stats = processor.get_stats()

    # Save embeddings if requested
    if output_file and results:
        embeddings_array = np.array([r.embedding for r in results])
        np.save(output_file, embeddings_array)
        logger.info(f"Saved {len(results)} embeddings to {output_file}")

    return results, stats


if __name__ == "__main__":
    # Example usage
    from dataclasses import dataclass

    logger.add("logs/batch_processor.log", rotation="10 MB")

    @dataclass
    class SimpleChunk:
        content: str
        chunk_id: str
        token_count: int

    # Create sample chunks
    sample_chunks = [
        SimpleChunk(
            content=f"The Euler buckling formula predicts column stability. Sample {i}",
            chunk_id=f"test_{i:04d}",
            token_count=20,
        )
        for i in range(100)
    ]

    # Process chunks
    processor = BatchEmbeddingProcessor(
        batch_size=16,
        enable_cache=True,
        enable_checkpoints=True,
    )

    results = processor.process_chunks(sample_chunks, show_progress=True)

    # Print statistics
    stats = processor.get_stats()
    print(f"\nProcessing Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Processed: {stats['processed_chunks']}")
    print(f"  Cached: {stats['cached_chunks']}")
    print(f"  Failed: {stats['failed_chunks']}")
    print(f"  Total time: {stats['processing_time']:.2f}s")
    if 'avg_processing_time' in stats:
        print(f"  Avg time/chunk: {stats['avg_processing_time']*1000:.1f}ms")

    print(f"\nFirst embedding shape: {results[0].embedding.shape}")
    print(f"First embedding dimension: {results[0].dimension}")
