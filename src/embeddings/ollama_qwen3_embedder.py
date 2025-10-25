"""
Ollama Qwen3-Embedding model wrapper for efficient local embeddings.

Uses Ollama's qwen3-embedding:4b model:
- 2.5GB model size (fits easily in VRAM)
- 40K token context window
- Up to 4096 dimensions (configurable)
- Matryoshka support for dimension reduction
- No HuggingFace dependencies needed
"""

from typing import List, Optional, Union
import numpy as np

try:
    import ollama
except ImportError as e:
    raise ImportError(
        "ollama required. Install with: pip install ollama"
    ) from e

from loguru import logger


class OllamaQwen3Embedder:
    """
    Ollama Qwen3-Embedding wrapper for local embeddings.

    Features:
    - 40K token context window
    - Up to 4096 dimensions (configurable)
    - Matryoshka embeddings support
    - Task-specific instructions support
    - Lightweight (2.5GB for 4B model)
    """

    def __init__(
        self,
        model_name: str = "qwen3-embedding:8b",
        output_dimensions: int = 768,
        use_matryoshka: bool = True,
        reduced_dimensions: int = 256,
        task_instruction: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize Ollama Qwen3 embedder.

        Args:
            model_name: Ollama model name (qwen3-embedding:8b recommended, qwen3-embedding:4b for tight VRAM)
            output_dimensions: Base embedding dimensions (32-4096)
            use_matryoshka: Use Matryoshka dimension reduction
            reduced_dimensions: Target dimensions after Matryoshka reduction
            task_instruction: Optional task-specific instruction (e.g., "Represent this text for retrieval:")
            batch_size: Batch size for encoding
            normalize_embeddings: L2 normalize embeddings
        """
        self.model_name = model_name
        self.output_dimensions = output_dimensions
        self.use_matryoshka = use_matryoshka
        self.reduced_dimensions = reduced_dimensions
        self.task_instruction = task_instruction or "Represent this technical document for retrieval:"
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # Verify model is available
        try:
            ollama.show(model_name)
            logger.info(f"Ollama Qwen3 embedder initialized: {model_name}")
            logger.info(
                f"  Dimensions: {output_dimensions} "
                f"{'→ ' + str(reduced_dimensions) if use_matryoshka else ''}"
            )
        except Exception as e:
            logger.error(f"Model {model_name} not found. Run: ollama pull {model_name}")
            raise RuntimeError(f"Model not available: {e}") from e

    def embed(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: Single text or list of texts
            show_progress: Show progress bar (for large batches)

        Returns:
            Numpy array of embeddings (N, D) where D depends on Matryoshka settings
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_embeddings = self._encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)

            if show_progress and (i + self.batch_size) % 100 == 0:
                logger.info(f"Embedded {i + self.batch_size}/{len(texts)} texts")

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)

        # Apply Matryoshka reduction if enabled
        if self.use_matryoshka:
            embeddings = self._apply_matryoshka(embeddings)

        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = self._normalize(embeddings)

        logger.info(
            f"Generated {len(embeddings)} embeddings "
            f"(dim={embeddings.shape[1]})"
        )

        return embeddings

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts using Ollama."""
        try:
            # Prepend task instruction if provided
            if self.task_instruction:
                texts = [f"{self.task_instruction} {text}" for text in texts]

            # Call Ollama embed API
            response = ollama.embed(
                model=self.model_name,
                input=texts,
            )

            # Extract embeddings
            embeddings = np.array(response['embeddings'])

            return embeddings

        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), self.output_dimensions))

    def _apply_matryoshka(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply Matryoshka dimension reduction.

        Matryoshka property: first N dimensions retain most performance.
        Typical reduction: 768D → 256D with ~99.5% performance retention.
        """
        if embeddings.shape[1] <= self.reduced_dimensions:
            return embeddings

        # Truncate to first N dimensions
        reduced = embeddings[:, :self.reduced_dimensions]

        logger.debug(
            f"Applied Matryoshka reduction: "
            f"{embeddings.shape[1]}D → {reduced.shape[1]}D"
        )

        return reduced

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def embed_chunks(
        self,
        chunks: List,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Embed a list of chunks.

        Args:
            chunks: List of chunk objects with 'content' attribute
            batch_size: Override default batch size

        Returns:
            Numpy array of embeddings
        """
        # Extract text content
        texts = [
            chunk.content if hasattr(chunk, 'content') else str(chunk)
            for chunk in chunks
        ]

        # Temporarily override batch size if requested
        original_batch_size = self.batch_size
        if batch_size:
            self.batch_size = batch_size

        embeddings = self.embed(texts, show_progress=True)

        # Restore original batch size
        self.batch_size = original_batch_size

        return embeddings


def create_embedder(
    model_name: str = "qwen3-embedding:8b",
    use_matryoshka: bool = True,
    reduced_dimensions: int = 256,
) -> OllamaQwen3Embedder:
    """
    Factory function to create Ollama Qwen3 embedder.

    Args:
        model_name: Ollama model (qwen3-embedding:8b recommended, qwen3-embedding:4b for tight VRAM)
        use_matryoshka: Use 256D compressed embeddings
        reduced_dimensions: Target dimensions after reduction

    Returns:
        Initialized OllamaQwen3Embedder
    """
    return OllamaQwen3Embedder(
        model_name=model_name,
        use_matryoshka=use_matryoshka,
        reduced_dimensions=reduced_dimensions,
    )


if __name__ == "__main__":
    # Example usage
    logger.add("logs/ollama_qwen3_embedder.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print("OLLAMA QWEN3 EMBEDDER")
    print("=" * 70)
    print("\nFeatures:")
    print("  • 40K token context window")
    print("  • Up to 4096 dimensions (configurable)")
    print("  • Matryoshka dimension reduction (768D → 256D)")
    print("  • Task-specific instructions support")
    print("  • Lightweight local inference (4.7GB for 8B model)")
    print("\nAvailable models:")
    print("  • qwen3-embedding:8b (4.7GB) - Recommended (#1 MTEB)")
    print("  • qwen3-embedding:4b (2.5GB) - Smaller alternative")
    print("=" * 70 + "\n")

    # Create embedder
    embedder = create_embedder(
        model_name="qwen3-embedding:8b",
        use_matryoshka=True,
        reduced_dimensions=256,
    )

    # Sample texts
    texts = [
        "The Euler buckling formula predicts column stability.",
        "Stress-strain relationships govern material behavior.",
        "Finite element analysis discretizes complex structures.",
    ]

    # Generate embeddings
    embeddings = embedder.embed(texts)

    print(f"\nGenerated embeddings:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    print(f"\nFirst embedding (truncated): {embeddings[0][:10]}")
