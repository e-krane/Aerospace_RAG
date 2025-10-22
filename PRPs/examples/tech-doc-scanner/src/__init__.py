"""
Technical Document Scanner - High-quality PDF to Markdown conversion for technical documents.

This package provides a production-ready application for converting technical PDFs
to Markdown with state-of-the-art equation recognition using Docling and Pix2Text.

Key Features:
- Docling: Structure detection, table extraction, OCR
- Pix2Text: SOTA mathematical formula recognition (MFR 1.5)
- LaTeX validation and cleaning with OCR fallback
- GPU acceleration support
- Batch processing with model caching
- Unified CLI and Python API

Usage:
    Command Line:
        tech-doc-scanner convert input.pdf --output-dir ./output
        tech-doc-scanner batch ./pdfs/*.pdf --output-dir ./outputs

    Python API:
        from tech_doc_scanner import DocumentConverterWrapper, DoclingConfig, Pix2TextConfig

        converter = DocumentConverterWrapper("pix2text", docling_config, pix2text_config)
        result = converter.convert_file("input.pdf", "./output")
"""

__version__ = "0.1.0"
__author__ = "docling-testing project"

# Public API imports
from .config import Config, DoclingConfig, OutputConfig, Pix2TextConfig
from .converter import ConversionResult, DocumentConverterWrapper
from .stats import BatchStatistics, ConversionStatistics

__all__ = [
    "__version__",
    "__author__",
    # Configuration
    "Config",
    "DoclingConfig",
    "Pix2TextConfig",
    "OutputConfig",
    # Conversion
    "DocumentConverterWrapper",
    "ConversionResult",
    # Statistics
    "ConversionStatistics",
    "BatchStatistics",
]
