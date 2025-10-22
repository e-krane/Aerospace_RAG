"""
Enrichment models for document processing.

This module provides custom enrichment models that extend Docling's
pipeline with additional processing capabilities.
"""

from .pix2text import Pix2TextFormulaEnrichmentModel

__all__ = ["Pix2TextFormulaEnrichmentModel"]
