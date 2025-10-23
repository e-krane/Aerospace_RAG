"""
Embedding quality validation tests.

Validates:
- Technical concept clustering
- Dissimilarity between unrelated concepts
- Mathematical equation semantic grouping
- Cross-language consistency
- Embedding space visualization (t-SNE/UMAP)
- Baseline comparison with OpenAI ada-002 (optional)
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings.qwen3_embedder import Qwen3Embedder, create_embedder


class TestEmbeddingQuality:
    """Test suite for embedding quality validation."""

    @pytest.fixture(scope="class")
    def embedder(self):
        """Create embedder instance for tests."""
        return create_embedder(use_matryoshka=True)

    @pytest.fixture(scope="class")
    def sample_texts(self):
        """Sample texts for testing."""
        return {
            # Similar technical concepts (structural mechanics)
            "buckling_1": "The Euler buckling formula predicts the critical load for column instability.",
            "buckling_2": "Column buckling occurs when the axial load exceeds the critical Euler load.",
            "buckling_3": "Slender columns fail by buckling under compressive loads.",

            # Similar equations (same concept, different notation)
            "stress_1": "Stress is defined as σ = F/A where F is force and A is area.",
            "stress_2": "The stress formula is force divided by area: σ = F/A.",
            "stress_3": "Normal stress equals the applied force over the cross-sectional area.",

            # Dissimilar concepts (unrelated topics)
            "chemistry": "The pH scale measures the acidity or alkalinity of a solution.",
            "history": "The French Revolution began in 1789 with the storming of the Bastille.",
            "biology": "Mitochondria are the powerhouse of the cell, producing ATP energy.",

            # Mathematical equations (should cluster)
            "eq_quadratic": "The quadratic formula is x = (-b ± √(b²-4ac)) / 2a.",
            "eq_euler": "Euler's formula states that e^(iπ) + 1 = 0.",
            "eq_pythagoras": "The Pythagorean theorem: a² + b² = c².",

            # Cross-language (English/Spanish structural mechanics)
            "beam_en": "A simply supported beam deflects under uniform load.",
            "beam_es": "Una viga simplemente apoyada se deflecta bajo carga uniforme.",

            # Material properties (should cluster)
            "material_1": "The elastic modulus relates stress to strain in linear materials.",
            "material_2": "Young's modulus describes material stiffness.",
            "material_3": "The modulus of elasticity is a measure of material rigidity.",
        }

    def test_similar_concepts_cluster(self, embedder, sample_texts):
        """
        Test that similar technical concepts have high cosine similarity.

        Validates that buckling-related texts cluster together.
        """
        # Embed buckling texts
        buckling_texts = [
            sample_texts["buckling_1"],
            sample_texts["buckling_2"],
            sample_texts["buckling_3"],
        ]

        embeddings = embedder.embed(buckling_texts)

        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        avg_similarity = np.mean(similarities)

        # Similar concepts should have >0.7 similarity
        assert avg_similarity > 0.7, (
            f"Similar concepts should cluster (avg similarity={avg_similarity:.3f})"
        )

        print(f"\n✓ Buckling concept cluster: {avg_similarity:.3f} avg similarity")

    def test_equation_semantic_grouping(self, embedder, sample_texts):
        """
        Test that semantically similar equations group together.

        Validates that stress formula variations cluster.
        """
        stress_texts = [
            sample_texts["stress_1"],
            sample_texts["stress_2"],
            sample_texts["stress_3"],
        ]

        embeddings = embedder.embed(stress_texts)

        # Calculate similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        avg_similarity = np.mean(similarities)

        # Equation variations should have >0.75 similarity
        assert avg_similarity > 0.75, (
            f"Equation variations should cluster (avg similarity={avg_similarity:.3f})"
        )

        print(f"✓ Stress equation cluster: {avg_similarity:.3f} avg similarity")

    def test_dissimilar_concepts_low_similarity(self, embedder, sample_texts):
        """
        Test that unrelated concepts have low similarity.

        Validates that chemistry, history, and biology texts are dissimilar.
        """
        dissimilar_texts = [
            sample_texts["chemistry"],
            sample_texts["history"],
            sample_texts["biology"],
        ]

        embeddings = embedder.embed(dissimilar_texts)

        # Calculate similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        avg_similarity = np.mean(similarities)

        # Unrelated concepts should have <0.4 similarity
        assert avg_similarity < 0.4, (
            f"Dissimilar concepts should have low similarity (avg={avg_similarity:.3f})"
        )

        print(f"✓ Dissimilar concepts: {avg_similarity:.3f} avg similarity (low)")

    def test_cross_language_consistency(self, embedder, sample_texts):
        """
        Test that equivalent concepts in different languages are similar.

        Validates English/Spanish translation consistency.
        """
        en_text = sample_texts["beam_en"]
        es_text = sample_texts["beam_es"]

        embeddings = embedder.embed([en_text, es_text])

        similarity = self._cosine_similarity(embeddings[0], embeddings[1])

        # Cross-language should have >0.6 similarity
        assert similarity > 0.6, (
            f"Cross-language consistency should be high (similarity={similarity:.3f})"
        )

        print(f"✓ Cross-language (EN/ES): {similarity:.3f} similarity")

    def test_material_properties_cluster(self, embedder, sample_texts):
        """
        Test that material property descriptions cluster together.
        """
        material_texts = [
            sample_texts["material_1"],
            sample_texts["material_2"],
            sample_texts["material_3"],
        ]

        embeddings = embedder.embed(material_texts)

        # Calculate similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        avg_similarity = np.mean(similarities)

        # Material properties should cluster
        assert avg_similarity > 0.7, (
            f"Material properties should cluster (avg similarity={avg_similarity:.3f})"
        )

        print(f"✓ Material properties cluster: {avg_similarity:.3f} avg similarity")

    def test_embedding_dimensions(self, embedder, sample_texts):
        """Test that embeddings have correct dimensions."""
        text = sample_texts["buckling_1"]
        embedding = embedder.embed(text)

        # Should be 256D with Matryoshka
        assert embedding.shape == (1, 256), (
            f"Expected (1, 256), got {embedding.shape}"
        )

        print(f"✓ Embedding dimensions: {embedding.shape}")

    def test_embedding_normalization(self, embedder, sample_texts):
        """Test that embeddings are L2 normalized."""
        text = sample_texts["buckling_1"]
        embedding = embedder.embed(text)[0]

        # Calculate L2 norm
        norm = np.linalg.norm(embedding)

        # Should be approximately 1.0 (normalized)
        assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: norm={norm:.3f}"

        print(f"✓ Embedding normalization: L2 norm={norm:.6f}")

    def test_batch_consistency(self, embedder, sample_texts):
        """
        Test that batch and individual embeddings are consistent.
        """
        texts = [sample_texts["buckling_1"], sample_texts["stress_1"]]

        # Batch embedding
        batch_embeddings = embedder.embed(texts)

        # Individual embeddings
        individual_embeddings = [embedder.embed(text) for text in texts]

        # Compare
        for i, (batch_emb, ind_emb) in enumerate(zip(batch_embeddings, individual_embeddings)):
            similarity = self._cosine_similarity(batch_emb, ind_emb[0])
            assert similarity > 0.99, (
                f"Batch/individual mismatch for text {i}: similarity={similarity:.3f}"
            )

        print(f"✓ Batch consistency: embeddings match individual processing")

    def test_structural_vs_unrelated(self, embedder, sample_texts):
        """
        Test discrimination between structural mechanics and unrelated topics.
        """
        # Structural mechanics
        structural = sample_texts["buckling_1"]

        # Unrelated
        unrelated = sample_texts["history"]

        embeddings = embedder.embed([structural, unrelated])

        similarity = self._cosine_similarity(embeddings[0], embeddings[1])

        # Should have low similarity
        assert similarity < 0.3, (
            f"Structural vs unrelated should have low similarity (got {similarity:.3f})"
        )

        print(f"✓ Structural vs unrelated: {similarity:.3f} similarity (discriminative)")

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class TestEmbeddingVisualization:
    """Test embedding space visualization (optional, requires matplotlib)."""

    @pytest.fixture(scope="class")
    def embedder(self):
        """Create embedder instance."""
        return create_embedder(use_matryoshka=True)

    def test_generate_embedding_visualization(self, embedder):
        """
        Generate t-SNE/UMAP visualization of embedding space.

        This test is optional and skipped if visualization libraries
        are not available.
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("Visualization libraries not available")

        # Sample texts from different categories
        texts = {
            "Buckling": [
                "Column buckling under axial load",
                "Euler buckling formula for slender columns",
                "Critical buckling stress prediction",
            ],
            "Stress": [
                "Normal stress equals force over area",
                "Stress-strain relationship in materials",
                "Yield stress defines material failure",
            ],
            "Deflection": [
                "Beam deflection under distributed load",
                "Maximum deflection at beam midpoint",
                "Deflection formula for cantilever beams",
            ],
            "Unrelated": [
                "The pH scale measures acidity",
                "The French Revolution began in 1789",
                "Mitochondria produce ATP energy",
            ],
        }

        # Flatten texts and track categories
        all_texts = []
        categories = []

        for category, text_list in texts.items():
            all_texts.extend(text_list)
            categories.extend([category] * len(text_list))

        # Generate embeddings
        embeddings = embedder.embed(all_texts)

        # t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create visualization
        plt.figure(figsize=(10, 8))

        # Plot each category with different color
        colors = {"Buckling": "red", "Stress": "blue", "Deflection": "green", "Unrelated": "gray"}

        for category in texts.keys():
            mask = [c == category for c in categories]
            x = embeddings_2d[mask, 0]
            y = embeddings_2d[mask, 1]
            plt.scatter(x, y, c=colors[category], label=category, alpha=0.7, s=100)

        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.title("Embedding Space Visualization (t-SNE)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save visualization
        output_dir = Path(__file__).parent.parent / "outputs" / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "embedding_space_tsne.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        print(f"\n✓ Embedding visualization saved: {output_file}")

        # Verify clusters are separated
        # Calculate within-cluster and between-cluster distances
        within_distances = []
        between_distances = []

        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if i < j:
                    dist = np.linalg.norm(embeddings_2d[i] - embeddings_2d[j])

                    if cat1 == cat2 and cat1 != "Unrelated":
                        within_distances.append(dist)
                    elif cat1 != cat2 and cat1 != "Unrelated" and cat2 != "Unrelated":
                        between_distances.append(dist)

        avg_within = np.mean(within_distances) if within_distances else 0
        avg_between = np.mean(between_distances) if between_distances else 0

        print(f"✓ Within-cluster distance: {avg_within:.3f}")
        print(f"✓ Between-cluster distance: {avg_between:.3f}")

        # Between should be larger than within (good separation)
        assert avg_between > avg_within, (
            f"Clusters should be separated (within={avg_within:.3f}, between={avg_between:.3f})"
        )


class TestEmbeddingPerformance:
    """Test embedding performance characteristics."""

    @pytest.fixture(scope="class")
    def embedder(self):
        """Create embedder instance."""
        return create_embedder(use_matryoshka=True)

    def test_embedding_speed(self, embedder):
        """Test embedding generation speed."""
        import time

        # Generate sample text
        text = "The Euler buckling formula predicts column stability." * 20  # Long text

        # Measure time
        start = time.time()
        embedding = embedder.embed(text)
        elapsed = time.time() - start

        # Should be fast (<1s for single text)
        assert elapsed < 1.0, f"Embedding too slow: {elapsed:.3f}s"

        print(f"\n✓ Embedding speed: {elapsed*1000:.1f}ms for {len(text)} chars")

    def test_batch_efficiency(self, embedder):
        """Test that batch processing is efficient."""
        import time

        texts = ["Sample text about structural mechanics."] * 100

        # Measure batch time
        start = time.time()
        embeddings = embedder.embed(texts)
        batch_time = time.time() - start

        time_per_text = batch_time / len(texts)

        # Should be efficient (<100ms per text in batch)
        assert time_per_text < 0.1, (
            f"Batch processing inefficient: {time_per_text*1000:.1f}ms per text"
        )

        print(f"\n✓ Batch efficiency: {time_per_text*1000:.1f}ms per text ({len(texts)} texts)")


def test_matryoshka_compression():
    """
    Test Matryoshka dimension reduction.

    Validates that 256D compressed embeddings retain semantic meaning.
    """
    # Create embedders with and without Matryoshka
    embedder_full = Qwen3Embedder(use_matryoshka=False)
    embedder_compressed = Qwen3Embedder(use_matryoshka=True)

    text = "The Euler buckling formula predicts column stability."

    # Generate embeddings
    emb_full = embedder_full.embed(text)[0]
    emb_compressed = embedder_compressed.embed(text)[0]

    # Check dimensions
    assert len(emb_full) == 768, f"Full embedding should be 768D, got {len(emb_full)}"
    assert len(emb_compressed) == 256, f"Compressed should be 256D, got {len(emb_compressed)}"

    # Compressed should match first 256 dimensions of full
    # (Matryoshka property)
    similarity = np.dot(emb_compressed, emb_full[:256])

    assert similarity > 0.95, (
        f"Matryoshka should preserve semantic meaning (similarity={similarity:.3f})"
    )

    print(f"\n✓ Matryoshka compression: 768D → 256D (similarity={similarity:.3f})")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
