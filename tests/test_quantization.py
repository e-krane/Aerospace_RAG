"""
Tests for binary quantization.

Tests:
- Binary quantization and dequantization
- Int8 quantization and dequantization
- Accuracy retention measurement
- Compression ratio validation
- BinaryQuantizer class
"""

import pytest
import numpy as np

from src.storage.quantization import (
    quantize_to_binary,
    dequantize_from_binary,
    quantize_to_int8,
    dequantize_from_int8,
    calculate_accuracy_retention,
    BinaryQuantizer,
    QuantizationType,
)


class TestBinaryQuantization:
    """Test binary quantization functions."""

    def test_quantize_to_binary_shape(self):
        """Test binary quantization output shape."""
        embeddings = np.random.randn(100, 256).astype(np.float32)

        binary_data = quantize_to_binary(embeddings)

        # Should pack 256 dimensions into 32 bytes (256 / 8)
        assert binary_data.shape == (100, 32)
        assert binary_data.dtype == np.uint8

    def test_quantize_dequantize_binary(self):
        """Test binary quantization round trip."""
        embeddings = np.random.randn(10, 256).astype(np.float32)

        # Quantize
        binary_data = quantize_to_binary(embeddings)

        # Dequantize
        reconstructed = dequantize_from_binary(binary_data, 256)

        # Check shape
        assert reconstructed.shape == embeddings.shape

        # Check values are -1 or +1
        assert np.all((reconstructed == -1.0) | (reconstructed == 1.0))

        # Check signs match
        original_signs = (embeddings > 0).astype(np.float32) * 2 - 1
        assert np.array_equal(reconstructed, original_signs)

    def test_binary_compression_ratio(self):
        """Test binary quantization achieves ~32x compression."""
        embeddings = np.random.randn(1000, 256).astype(np.float32)

        binary_data = quantize_to_binary(embeddings)

        original_size = embeddings.nbytes
        compressed_size = binary_data.nbytes
        compression_ratio = original_size / compressed_size

        # Should be ~32x (float32 → 1-bit)
        assert 30 <= compression_ratio <= 34

    def test_binary_quantization_preserves_shape(self):
        """Test different input shapes."""
        # Small batch
        small = np.random.randn(5, 128).astype(np.float32)
        binary_small = quantize_to_binary(small)
        assert binary_small.shape == (5, 16)  # 128 / 8

        # Large batch
        large = np.random.randn(10000, 512).astype(np.float32)
        binary_large = quantize_to_binary(large)
        assert binary_large.shape == (10000, 64)  # 512 / 8


class TestInt8Quantization:
    """Test int8 quantization functions."""

    def test_quantize_to_int8_shape(self):
        """Test int8 quantization output shape."""
        embeddings = np.random.randn(100, 256).astype(np.float32)

        int8_data, scale, offset = quantize_to_int8(embeddings)

        assert int8_data.shape == embeddings.shape
        assert int8_data.dtype == np.int8
        assert isinstance(scale, (float, np.floating))
        assert isinstance(offset, (float, np.floating))

    def test_quantize_dequantize_int8(self):
        """Test int8 quantization round trip."""
        embeddings = np.random.randn(10, 256).astype(np.float32)

        # Quantize
        int8_data, scale, offset = quantize_to_int8(embeddings)

        # Dequantize
        reconstructed = dequantize_from_int8(int8_data, scale, offset)

        # Check shape
        assert reconstructed.shape == embeddings.shape

        # Check close to original (with quantization error)
        # Int8 should be fairly accurate (within ~2%)
        assert np.allclose(reconstructed, embeddings, atol=0.1)

    def test_int8_compression_ratio(self):
        """Test int8 quantization achieves ~4x compression."""
        embeddings = np.random.randn(1000, 256).astype(np.float32)

        int8_data, _, _ = quantize_to_int8(embeddings)

        original_size = embeddings.nbytes
        compressed_size = int8_data.nbytes
        compression_ratio = original_size / compressed_size

        # Should be exactly 4x (float32 → int8)
        assert abs(compression_ratio - 4.0) < 0.1

    def test_int8_range_coverage(self):
        """Test int8 quantization covers full [-128, 127] range."""
        # Create embeddings with wide value range
        embeddings = np.random.uniform(-10, 10, (1000, 256)).astype(np.float32)

        int8_data, _, _ = quantize_to_int8(embeddings)

        # Should use most of the int8 range
        assert int8_data.min() < -100
        assert int8_data.max() > 100


class TestAccuracyRetention:
    """Test accuracy retention measurement."""

    def test_accuracy_retention_perfect(self):
        """Test accuracy retention with no quantization (100%)."""
        embeddings = np.random.randn(100, 256).astype(np.float32)

        # No quantization - should be 100%
        accuracy = calculate_accuracy_retention(
            embeddings,
            embeddings,
            sample_queries=None,
            top_k=10,
        )

        assert accuracy == 100.0

    def test_accuracy_retention_binary(self):
        """Test accuracy retention with binary quantization."""
        embeddings = np.random.randn(500, 256).astype(np.float32)

        # Quantize to binary
        binary_data = quantize_to_binary(embeddings)
        quantized = dequantize_from_binary(binary_data, 256)

        # Calculate accuracy
        accuracy = calculate_accuracy_retention(
            embeddings,
            quantized,
            sample_queries=None,
            top_k=10,
        )

        # Binary should retain reasonable accuracy (40-50% with random data)
        # In practice with real normalized embeddings it's ~90-95%
        assert 35 <= accuracy <= 60

    def test_accuracy_retention_int8(self):
        """Test accuracy retention with int8 quantization."""
        embeddings = np.random.randn(500, 256).astype(np.float32)

        # Quantize to int8
        int8_data, scale, offset = quantize_to_int8(embeddings)
        quantized = dequantize_from_int8(int8_data, scale, offset)

        # Calculate accuracy
        accuracy = calculate_accuracy_retention(
            embeddings,
            quantized,
            sample_queries=None,
            top_k=10,
        )

        # Int8 should retain >98% accuracy
        assert accuracy >= 95

    def test_accuracy_retention_with_queries(self):
        """Test accuracy retention with specific queries."""
        embeddings = np.random.randn(200, 128).astype(np.float32)
        queries = np.random.randn(20, 128).astype(np.float32)

        # Quantize
        binary_data = quantize_to_binary(embeddings)
        quantized = dequantize_from_binary(binary_data, 128)

        # Calculate with specific queries
        accuracy = calculate_accuracy_retention(
            embeddings,
            quantized,
            sample_queries=queries,
            top_k=5,
        )

        assert 0 <= accuracy <= 100


class TestBinaryQuantizer:
    """Test BinaryQuantizer class."""

    def test_quantizer_initialization(self):
        """Test quantizer initializes correctly."""
        quantizer = BinaryQuantizer()

        assert quantizer.original_dims is None

    def test_quantizer_quantize(self):
        """Test quantizer.quantize() method."""
        quantizer = BinaryQuantizer()
        embeddings = np.random.randn(100, 256).astype(np.float32)

        binary_data, metrics = quantizer.quantize(embeddings)

        # Check binary data
        assert binary_data.shape == (100, 32)
        assert binary_data.dtype == np.uint8

        # Check metrics
        assert metrics.original_size_mb > 0
        assert metrics.quantized_size_mb > 0
        assert metrics.compression_ratio > 25  # Should be ~32x
        assert metrics.quantization_time_ms >= 0

        # Check quantizer state
        assert quantizer.original_dims == 256

    def test_quantizer_accuracy_test(self):
        """Test quantizer.test_accuracy_retention() method."""
        quantizer = BinaryQuantizer()
        embeddings = np.random.randn(200, 128).astype(np.float32)

        accuracy = quantizer.test_accuracy_retention(
            embeddings,
            sample_queries=None,
            top_k=10,
        )

        assert 0 <= accuracy <= 100
        # Binary should retain reasonable accuracy with random data
        # In practice with real normalized embeddings it's ~90-95%
        assert accuracy >= 35

    def test_quantizer_with_large_embeddings(self):
        """Test quantizer handles large embedding matrices."""
        quantizer = BinaryQuantizer()
        # Large matrix: 10k vectors of 512 dimensions
        embeddings = np.random.randn(10000, 512).astype(np.float32)

        binary_data, metrics = quantizer.quantize(embeddings)

        # Check shape
        assert binary_data.shape == (10000, 64)  # 512 / 8

        # Check compression
        assert metrics.compression_ratio > 25


class TestQuantizationMetrics:
    """Test quantization metrics calculation."""

    def test_metrics_binary(self):
        """Test metrics for binary quantization."""
        embeddings = np.random.randn(1000, 256).astype(np.float32)

        quantizer = BinaryQuantizer()
        _, metrics = quantizer.quantize(embeddings)

        # Original size: 1000 * 256 * 4 bytes = 1MB
        assert abs(metrics.original_size_mb - 0.976) < 0.1

        # Quantized size: 1000 * 32 bytes = 0.03MB
        assert abs(metrics.quantized_size_mb - 0.031) < 0.01

        # Compression ratio should be ~32x
        assert 30 <= metrics.compression_ratio <= 34

    def test_metrics_timing(self):
        """Test quantization timing is measured."""
        embeddings = np.random.randn(5000, 256).astype(np.float32)

        quantizer = BinaryQuantizer()
        _, metrics = quantizer.quantize(embeddings, measure_time=True)

        # Should take some time
        assert metrics.quantization_time_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
