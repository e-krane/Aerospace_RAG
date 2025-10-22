"""Tests for configuration module."""

import pytest
from pathlib import Path

from tech_doc_scanner.config import Config, DoclingConfig, Pix2TextConfig


class TestDoclingConfig:
    """Test suite for DoclingConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = DoclingConfig()
        assert config.do_ocr is True
        assert config.ocr_languages == ["en"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = DoclingConfig()
        data = config.to_dict()
        assert isinstance(data, dict)
        assert "do_ocr" in data


class TestPix2TextConfig:
    """Test suite for Pix2TextConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = Pix2TextConfig()
        assert config.enabled is True
        assert config.validate is True


class TestConfig:
    """Test suite for Config."""

    def test_default_values(self):
        """Test that default Config is created correctly."""
        config = Config()
        assert isinstance(config.docling, DoclingConfig)
        assert isinstance(config.pix2text, Pix2TextConfig)
