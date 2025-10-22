"""End-to-end integration tests."""

import pytest
from pathlib import Path
import tempfile
from tech_doc_scanner.converter import DocumentConverterWrapper
from tech_doc_scanner.config import DoclingConfig, Pix2TextConfig, OutputConfig


class TestEndToEndIntegration:
    """End-to-end integration tests with real PDF processing."""

    @pytest.fixture
    def test_pdf(self):
        """Path to test PDF (short Bruhn document)."""
        pdf_path = Path("Bruhn_Crippling_short.pdf")
        if not pdf_path.exists():
            pytest.skip("Test PDF not found: Bruhn_Crippling_short.pdf")
        return pdf_path

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_standard_pipeline_conversion(self, test_pdf, temp_output_dir):
        """Test end-to-end conversion with standard pipeline."""
        wrapper = DocumentConverterWrapper(
            pipeline_type="standard",
            docling_config=DoclingConfig(),
            pix2text_config=Pix2TextConfig(),
            output_config=OutputConfig(base_dir=temp_output_dir, formats=["md"]),
        )
        
        result = wrapper.convert_file(test_pdf, temp_output_dir)
        
        assert result.success is True
        assert result.error is None
        assert len(result.output_files) > 0
        assert result.elapsed_time > 0
        
        # Check markdown file was created
        md_file = temp_output_dir / f"{test_pdf.stem}.md"
        assert md_file.exists()
        assert md_file.stat().st_size > 0

    @pytest.mark.slow
    def test_pix2text_pipeline_conversion(self, test_pdf, temp_output_dir):
        """Test end-to-end conversion with pix2text pipeline (slower)."""
        wrapper = DocumentConverterWrapper(
            pipeline_type="pix2text",
            docling_config=DoclingConfig(),
            pix2text_config=Pix2TextConfig(device="cpu", validate=True, fallback_to_ocr=True),
            output_config=OutputConfig(base_dir=temp_output_dir, formats=["md", "html"]),
        )
        
        result = wrapper.convert_file(test_pdf, temp_output_dir)
        
        assert result.success is True
        assert result.error is None
        assert len(result.output_files) >= 2  # md + html
        assert result.elapsed_time > 0
        
        # Check both output files
        md_file = temp_output_dir / f"{test_pdf.stem}.md"
        html_file = temp_output_dir / f"{test_pdf.stem}.html"
        assert md_file.exists()
        assert html_file.exists()
        
        # Check formula stats if available
        if result.stats:
            assert result.stats.total >= 0
            assert result.stats.recognized >= 0
            assert 0 <= result.stats.success_rate <= 100

    def test_nonexistent_file_handling(self, temp_output_dir):
        """Test error handling for nonexistent file."""
        wrapper = DocumentConverterWrapper(
            pipeline_type="standard",
            docling_config=DoclingConfig(),
            pix2text_config=Pix2TextConfig(),
            output_config=OutputConfig(base_dir=temp_output_dir),
        )
        
        fake_pdf = Path("nonexistent.pdf")
        result = wrapper.convert_file(fake_pdf, temp_output_dir)
        
        # Should handle error gracefully
        assert result.success is False
        assert result.error is not None
        assert len(result.output_files) == 0

    def test_config_yaml_roundtrip(self, temp_output_dir):
        """Test YAML config save and load."""
        from tech_doc_scanner.config import Config
        
        # Create config
        config = Config(
            docling=DoclingConfig(do_ocr=True, ocr_languages=["en"]),
            pix2text=Pix2TextConfig(device="cpu", validate=True),
            output=OutputConfig(formats=["md", "html"]),
        )
        
        # Save to YAML
        config_path = temp_output_dir / "test_config.yaml"
        config.to_yaml(config_path)
        assert config_path.exists()
        
        # Load from YAML
        loaded_config = Config.from_yaml(config_path)
        assert loaded_config.docling.do_ocr is True
        assert loaded_config.pix2text.device == "cpu"
        assert "md" in loaded_config.output.formats
