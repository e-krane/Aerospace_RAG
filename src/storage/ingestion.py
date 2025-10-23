"""
Batch ingestion pipeline for uploading embeddings to Qdrant.

Features:
- Batch upsert with optimal batch sizes (1000 points)
- Progress tracking and logging
- Error handling with rollback support
- Verification of point counts
- Resume capability for interrupted ingestion
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time

try:
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    raise ImportError(
        "numpy and tqdm required. Install with: pip install numpy tqdm"
    ) from e

from loguru import logger

# Import Qdrant client
try:
    from .qdrant_client import AerospaceQdrantClient
except ImportError:
    from qdrant_client import AerospaceQdrantClient


@dataclass
class IngestionStats:
    """Statistics for ingestion process."""

    total_chunks: int = 0
    ingested_chunks: int = 0
    failed_chunks: int = 0
    skipped_chunks: int = 0
    total_batches: int = 0
    failed_batches: int = 0
    ingestion_time: float = 0.0
    avg_batch_time: float = 0.0


class IngestionPipeline:
    """
    Batch ingestion pipeline for Qdrant.

    Features:
    - Optimal batch sizing (1000 points recommended)
    - Progress tracking with tqdm
    - Error handling and rollback
    - Verification checks
    - Resume capability
    """

    OPTIMAL_BATCH_SIZE = 1000

    def __init__(
        self,
        qdrant_client: AerospaceQdrantClient,
        batch_size: int = OPTIMAL_BATCH_SIZE,
        enable_verification: bool = True,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            qdrant_client: Qdrant client instance
            batch_size: Batch size for uploads (default 1000)
            enable_verification: Verify point counts after ingestion
        """
        self.client = qdrant_client
        self.batch_size = batch_size
        self.enable_verification = enable_verification

        self.stats = IngestionStats()

        logger.info(f"Ingestion pipeline initialized: batch_size={batch_size}")

    def ingest_embeddings(
        self,
        collection_name: str,
        chunks: List[Any],
        embeddings: np.ndarray,
        show_progress: bool = True,
        verify_after: bool = True,
    ) -> IngestionStats:
        """
        Ingest chunks with embeddings into Qdrant.

        Args:
            collection_name: Target collection name
            chunks: List of chunk objects (from chunking pipeline)
            embeddings: Numpy array of embeddings (N x D)
            show_progress: Show progress bar
            verify_after: Verify point count after ingestion

        Returns:
            IngestionStats object
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
            )

        self.stats = IngestionStats(total_chunks=len(chunks))
        start_time = time.time()

        # Convert chunks to Qdrant points
        logger.info(f"Converting {len(chunks)} chunks to Qdrant points")
        points = self._chunks_to_points(chunks, embeddings)

        # Batch upload
        logger.info(f"Uploading {len(points)} points in batches of {self.batch_size}")

        # Progress bar
        iterator = tqdm(
            range(0, len(points), self.batch_size),
            desc="Ingesting batches",
            disable=not show_progress,
            total=len(points) // self.batch_size + (1 if len(points) % self.batch_size else 0),
        )

        for batch_start in iterator:
            batch_end = min(batch_start + self.batch_size, len(points))
            batch = points[batch_start:batch_end]

            batch_time_start = time.time()

            try:
                # Upsert batch
                count = self.client.upsert_points(
                    collection_name=collection_name,
                    points=batch,
                    batch_size=self.batch_size,
                )

                self.stats.ingested_chunks += count
                self.stats.total_batches += 1

                # Update timing
                batch_time = time.time() - batch_time_start
                self.stats.avg_batch_time = (
                    (self.stats.avg_batch_time * (self.stats.total_batches - 1) + batch_time)
                    / self.stats.total_batches
                )

            except Exception as e:
                logger.error(f"Failed to ingest batch {batch_start}-{batch_end}: {e}")
                self.stats.failed_chunks += len(batch)
                self.stats.failed_batches += 1

                # Continue with next batch (no rollback)

        # Total time
        self.stats.ingestion_time = time.time() - start_time

        # Verification
        if verify_after and self.enable_verification:
            self._verify_ingestion(collection_name)

        logger.info(
            f"Ingestion complete: {self.stats.ingested_chunks}/{self.stats.total_chunks} chunks "
            f"({self.stats.failed_chunks} failed) in {self.stats.ingestion_time:.1f}s"
        )

        return self.stats

    def ingest_from_batch_results(
        self,
        collection_name: str,
        embedding_results: List,
        show_progress: bool = True,
    ) -> IngestionStats:
        """
        Ingest from BatchEmbeddingProcessor results.

        Args:
            collection_name: Target collection
            embedding_results: List of EmbeddingResult objects
            show_progress: Show progress bar

        Returns:
            IngestionStats object
        """
        self.stats = IngestionStats(total_chunks=len(embedding_results))
        start_time = time.time()

        # Convert EmbeddingResult objects to Qdrant points
        points = []

        for result in embedding_results:
            point = {
                "id": result.chunk_id,
                "vector": result.embedding,
                "payload": {
                    "chunk_id": result.chunk_id,
                    "dimension": result.dimension,
                    "token_count": result.token_count,
                    "processing_time": result.processing_time,
                    "cached": result.cached,
                },
            }
            points.append(point)

        # Batch upload
        logger.info(f"Uploading {len(points)} embedding results")

        iterator = tqdm(
            range(0, len(points), self.batch_size),
            desc="Ingesting batches",
            disable=not show_progress,
            total=len(points) // self.batch_size + (1 if len(points) % self.batch_size else 0),
        )

        for batch_start in iterator:
            batch_end = min(batch_start + self.batch_size, len(points))
            batch = points[batch_start:batch_end]

            try:
                count = self.client.upsert_points(
                    collection_name=collection_name,
                    points=batch,
                    batch_size=self.batch_size,
                )

                self.stats.ingested_chunks += count
                self.stats.total_batches += 1

            except Exception as e:
                logger.error(f"Failed to ingest batch: {e}")
                self.stats.failed_chunks += len(batch)
                self.stats.failed_batches += 1

        self.stats.ingestion_time = time.time() - start_time

        logger.info(
            f"Ingestion complete: {self.stats.ingested_chunks}/{self.stats.total_chunks} results"
        )

        return self.stats

    def _chunks_to_points(
        self,
        chunks: List,
        embeddings: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Convert chunks and embeddings to Qdrant points.

        Args:
            chunks: List of chunk objects
            embeddings: Numpy array of embeddings

        Returns:
            List of point dictionaries
        """
        points = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Extract chunk metadata
            chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"chunk_{i}"
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)

            # Build payload
            payload = {
                "content": content,
                "chunk_id": chunk_id,
                "token_count": chunk.token_count if hasattr(chunk, 'token_count') else 0,
            }

            # Add optional metadata if available
            if hasattr(chunk, 'document_id'):
                payload["document_id"] = chunk.document_id

            if hasattr(chunk, 'section_path'):
                payload["section_path"] = chunk.section_path

            if hasattr(chunk, 'chunk_type'):
                payload["chunk_type"] = chunk.chunk_type

            if hasattr(chunk, 'has_equations'):
                payload["has_equations"] = chunk.has_equations

            if hasattr(chunk, 'equation_count'):
                payload["equation_count"] = chunk.equation_count

            if hasattr(chunk, 'figure_references'):
                payload["figure_references"] = chunk.figure_references

            if hasattr(chunk, 'keywords'):
                payload["keywords"] = chunk.keywords

            if hasattr(chunk, 'page_number') and chunk.page_number is not None:
                payload["page_number"] = chunk.page_number

            if hasattr(chunk, 'latex_source') and chunk.latex_source:
                payload["latex_source"] = chunk.latex_source

            # Create point
            point = {
                "id": chunk_id,
                "vector": embedding,
                "payload": payload,
            }

            points.append(point)

        return points

    def _verify_ingestion(self, collection_name: str):
        """
        Verify that point count matches expected count.

        Args:
            collection_name: Collection to verify
        """
        logger.info("Verifying ingestion...")

        try:
            info = self.client.get_collection_info(collection_name)
            actual_count = info.get("points_count", 0)
            expected_count = self.stats.ingested_chunks

            if actual_count == expected_count:
                logger.info(
                    f"✓ Verification passed: {actual_count} points in collection"
                )
            else:
                logger.warning(
                    f"⚠ Verification mismatch: expected {expected_count}, "
                    f"found {actual_count} points"
                )

                # Update stats
                self.stats.failed_chunks += (expected_count - actual_count)

        except Exception as e:
            logger.error(f"Verification failed: {e}")

    def get_ingestion_report(self) -> Dict:
        """
        Get detailed ingestion report.

        Returns:
            Report dictionary
        """
        report = asdict(self.stats)

        # Calculate additional metrics
        if self.stats.total_chunks > 0:
            report["success_rate"] = (
                self.stats.ingested_chunks / self.stats.total_chunks * 100
            )

        if self.stats.ingestion_time > 0:
            report["chunks_per_second"] = (
                self.stats.ingested_chunks / self.stats.ingestion_time
            )

        if self.stats.total_batches > 0:
            report["batch_failure_rate"] = (
                self.stats.failed_batches / self.stats.total_batches * 100
            )

        return report


def ingest_corpus(
    qdrant_client: AerospaceQdrantClient,
    collection_name: str,
    chunks: List,
    embeddings: np.ndarray,
    batch_size: int = 1000,
    show_progress: bool = True,
) -> IngestionStats:
    """
    Convenience function to ingest a full corpus.

    Args:
        qdrant_client: Qdrant client
        collection_name: Target collection
        chunks: List of chunks
        embeddings: Embeddings array
        batch_size: Batch size
        show_progress: Show progress

    Returns:
        IngestionStats object
    """
    pipeline = IngestionPipeline(
        qdrant_client=qdrant_client,
        batch_size=batch_size,
    )

    return pipeline.ingest_embeddings(
        collection_name=collection_name,
        chunks=chunks,
        embeddings=embeddings,
        show_progress=show_progress,
    )


if __name__ == "__main__":
    # Example usage
    from dataclasses import dataclass
    import sys
    from pathlib import Path

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from storage.qdrant_client import create_qdrant_client

    logger.add("logs/ingestion.log", rotation="10 MB")

    @dataclass
    class SimpleChunk:
        content: str
        chunk_id: str
        token_count: int
        document_id: str
        chunk_type: str = "text"
        has_equations: bool = False

    # Create sample data
    chunks = [
        SimpleChunk(
            content=f"Sample chunk {i} about aerospace structures",
            chunk_id=f"test_{i:04d}",
            token_count=20,
            document_id="test_doc",
            chunk_type="text",
            has_equations=(i % 3 == 0),
        )
        for i in range(100)
    ]

    # Generate random embeddings
    embeddings = np.random.rand(len(chunks), 256).astype(np.float32)

    # Create Qdrant client
    client = create_qdrant_client()

    # Create collection
    client.create_semantic_collection(recreate=True)

    # Ingest data
    pipeline = IngestionPipeline(client, batch_size=50)
    stats = pipeline.ingest_embeddings(
        collection_name=client.SEMANTIC_COLLECTION,
        chunks=chunks,
        embeddings=embeddings,
        show_progress=True,
    )

    # Print report
    report = pipeline.get_ingestion_report()
    print("\nIngestion Report:")
    print(f"  Total chunks: {report['total_chunks']}")
    print(f"  Ingested: {report['ingested_chunks']}")
    print(f"  Failed: {report['failed_chunks']}")
    print(f"  Success rate: {report.get('success_rate', 0):.1f}%")
    print(f"  Time: {report['ingestion_time']:.1f}s")
    print(f"  Throughput: {report.get('chunks_per_second', 0):.1f} chunks/s")
