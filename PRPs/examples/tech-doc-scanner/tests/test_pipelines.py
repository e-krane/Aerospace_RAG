"""Integration tests for pipeline factory."""

import pytest
from tech_doc_scanner.pipelines import PipelineFactory, Pix2TextPipeline
from tech_doc_scanner.config import DoclingConfig, Pix2TextConfig
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


class TestPipelineFactory:
    """Test suite for PipelineFactory."""

    def test_create_standard_pipeline(self):
        """Test creating standard pipeline."""
        config = DoclingConfig()
        pipeline = PipelineFactory.create_standard_pipeline(config)
        assert pipeline is not None
        assert isinstance(pipeline, StandardPdfPipeline)

    def test_create_pix2text_pipeline(self):
        """Test creating pix2text pipeline."""
        docling_config = DoclingConfig()
        pix2text_config = Pix2TextConfig(device="cpu")
        pipeline = PipelineFactory.create_pix2text_pipeline(docling_config, pix2text_config)
        assert pipeline is not None
        assert isinstance(pipeline, Pix2TextPipeline)

    def test_create_pipeline_by_string_standard(self):
        """Test creating pipeline by type string (standard)."""
        docling_config = DoclingConfig()
        pix2text_config = Pix2TextConfig()
        pipeline = PipelineFactory.create_pipeline("standard", docling_config, pix2text_config)
        assert isinstance(pipeline, StandardPdfPipeline)

    def test_create_pipeline_by_string_pix2text(self):
        """Test creating pipeline by type string (pix2text)."""
        docling_config = DoclingConfig()
        pix2text_config = Pix2TextConfig(device="cpu")
        pipeline = PipelineFactory.create_pipeline("pix2text", docling_config, pix2text_config)
        assert isinstance(pipeline, Pix2TextPipeline)

    def test_create_pipeline_invalid_type(self):
        """Test that invalid pipeline type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid pipeline type"):
            PipelineFactory.create_pipeline("invalid", DoclingConfig(), Pix2TextConfig())

    def test_pix2text_pipeline_has_enrichment(self):
        """Test Pix2TextPipeline is created with formula enrichment."""
        docling_config = DoclingConfig()
        pix2text_config = Pix2TextConfig(device="cpu", validate=True, fallback_to_ocr=True)
        pipeline = PipelineFactory.create_pix2text_pipeline(docling_config, pix2text_config)
        assert pipeline is not None
        # Pipeline should be a Pix2TextPipeline instance
        assert isinstance(pipeline, Pix2TextPipeline)
