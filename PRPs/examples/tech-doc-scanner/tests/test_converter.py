"""Integration tests for DocumentConverterWrapper."""

import pytest
from pathlib import Path
from tech_doc_scanner.converter import DocumentConverterWrapper, ConversionResult
from tech_doc_scanner.config import DoclingConfig, Pix2TextConfig, OutputConfig


class TestDocumentConverterWrapper:
    """Test suite for DocumentConverterWrapper."""

    def test_initialization_standard(self):
        """Test wrapper initialization with standard pipeline."""
        wrapper = DocumentConverterWrapper(
            pipeline_type="standard",
            docling_config=DoclingConfig(),
            pix2text_config=Pix2TextConfig(),
            output_config=OutputConfig(),
        )
        assert wrapper is not None

    def test_initialization_pix2text(self):
        """Test wrapper initialization with pix2text pipeline."""
        wrapper = DocumentConverterWrapper(
            pipeline_type="pix2text",
            docling_config=DoclingConfig(),
            pix2text_config=Pix2TextConfig(device="cpu"),
            output_config=OutputConfig(),
        )
        assert wrapper is not None

    def test_conversion_result_dataclass(self):
        """Test ConversionResult dataclass creation."""
        result = ConversionResult(
            input_path=Path("test.pdf"),
            output_dir=Path("./output"),
            success=True,
            elapsed_time=10.5,
            stats=None,
            error=None,
            output_files=[Path("test.md")],
        )
        assert result.success is True
        assert result.input_path == Path("test.pdf")
        assert len(result.output_files) == 1
        assert result.elapsed_time == 10.5

    def test_conversion_result_failure(self):
        """Test ConversionResult for failed conversion."""
        result = ConversionResult(
            input_path=Path("bad.pdf"),
            output_dir=Path("./output"),
            success=False,
            elapsed_time=0.5,
            stats=None,
            error="File not found",
            output_files=[],
        )
        assert result.success is False
        assert result.error == "File not found"
        assert len(result.output_files) == 0

    def test_invalid_pipeline_type(self):
        """Test that invalid pipeline type raises ValueError."""
        with pytest.raises(ValueError):
            DocumentConverterWrapper(
                pipeline_type="invalid",
                docling_config=DoclingConfig(),
                pix2text_config=Pix2TextConfig(),
                output_config=OutputConfig(),
            )

    @pytest.mark.parametrize("pipeline_type", ["standard", "pix2text"])
    def test_wrapper_initialization_parametrized(self, pipeline_type):
        """Test wrapper initialization with different pipeline types."""
        pix2text_config = Pix2TextConfig(device="cpu") if pipeline_type == "pix2text" else Pix2TextConfig()
        wrapper = DocumentConverterWrapper(
            pipeline_type=pipeline_type,
            docling_config=DoclingConfig(),
            pix2text_config=pix2text_config,
            output_config=OutputConfig(),
        )
        assert wrapper is not None
