"""
Docling Parser for LaTeX technical documents.

This module provides a parser that converts LaTeX files to markdown while
preserving mathematical equations, document structure, and figure references.

Target performance: ~3.7 seconds/page
Target equation preservation: >95%
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
except ImportError as e:
    raise ImportError(
        "Docling is not installed. Install with: pip install docling"
    ) from e

from loguru import logger


@dataclass
class ParsedDocument:
    """Structured representation of a parsed document."""

    source_file: Path
    markdown_content: str
    title: str = ""
    chapter_number: Optional[int] = None
    sections: List[Dict] = field(default_factory=list)
    equations: List[Dict] = field(default_factory=list)
    figures: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    parsing_time_seconds: float = 0.0
    page_count: int = 0


class DoclingParser:
    """
    High-performance parser for LaTeX technical documents using Docling.

    Features:
    - Preserves LaTeX equation notation (>95% accuracy target)
    - Extracts document hierarchy (chapters, sections, subsections)
    - Identifies figure references and bounding boxes
    - Handles cross-references between chapters
    - Graceful error handling with detailed logging

    Performance:
    - Target: 3.7 seconds/page
    - Optimized for technical/scientific documents
    - GPU acceleration supported
    """

    def __init__(
        self,
        enable_formula_enrichment: bool = True,
        enable_figure_extraction: bool = True,
        enable_code_enrichment: bool = False,
        images_scale: int = 2,
        timeout_seconds: int = 300,
    ):
        """
        Initialize the Docling parser with optimal settings.

        Args:
            enable_formula_enrichment: Extract LaTeX from formulas (recommended for technical docs)
            enable_figure_extraction: Extract figure images and bounding boxes
            enable_code_enrichment: Parse code blocks (optional, increases processing time)
            images_scale: Scale factor for extracted images (1-4)
            timeout_seconds: Maximum time to wait for document conversion
        """
        self.enable_formula_enrichment = enable_formula_enrichment
        self.enable_figure_extraction = enable_figure_extraction
        self.enable_code_enrichment = enable_code_enrichment
        self.images_scale = images_scale
        self.timeout_seconds = timeout_seconds

        # Configure pipeline options
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_formula_enrichment = enable_formula_enrichment
        self.pipeline_options.do_code_enrichment = enable_code_enrichment

        if enable_figure_extraction:
            self.pipeline_options.generate_picture_images = True
            self.pipeline_options.images_scale = images_scale

        # Initialize converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )

        logger.info(
            f"Docling parser initialized: "
            f"formula_enrichment={enable_formula_enrichment}, "
            f"figure_extraction={enable_figure_extraction}, "
            f"code_enrichment={enable_code_enrichment}"
        )

    def parse_file(self, file_path: Path) -> ParsedDocument:
        """
        Parse a single LaTeX/PDF file to markdown with preserved structure.

        Args:
            file_path: Path to the LaTeX or PDF file to parse

        Returns:
            ParsedDocument with markdown content and extracted metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
            RuntimeError: If parsing fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Parsing file: {file_path}")

        try:
            import time
            start_time = time.time()

            # Convert document
            result = self.converter.convert(str(file_path))
            doc = result.document

            # Extract markdown
            markdown_content = doc.export_to_markdown()

            # Extract metadata
            title = self._extract_title(doc)
            sections = self._extract_sections(doc)
            equations = self._extract_equations(doc)
            figures = self._extract_figures(doc)

            # Calculate parsing time
            parsing_time = time.time() - start_time

            # Create parsed document
            parsed_doc = ParsedDocument(
                source_file=file_path,
                markdown_content=markdown_content,
                title=title,
                sections=sections,
                equations=equations,
                figures=figures,
                metadata={
                    "file_size_bytes": file_path.stat().st_size,
                    "parser": "docling",
                    "formula_enrichment": self.enable_formula_enrichment,
                    "figure_extraction": self.enable_figure_extraction,
                },
                parsing_time_seconds=parsing_time,
                page_count=getattr(doc, "num_pages", 0),
            )

            # Log performance
            if parsed_doc.page_count > 0:
                seconds_per_page = parsing_time / parsed_doc.page_count
                logger.info(
                    f"Parsed {file_path.name}: "
                    f"{parsed_doc.page_count} pages in {parsing_time:.2f}s "
                    f"({seconds_per_page:.2f}s/page), "
                    f"{len(equations)} equations, {len(figures)} figures"
                )
            else:
                logger.info(
                    f"Parsed {file_path.name}: {parsing_time:.2f}s, "
                    f"{len(equations)} equations, {len(figures)} figures"
                )

            return parsed_doc

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            raise RuntimeError(f"Parsing failed for {file_path}: {e}") from e

    def parse_batch(
        self,
        file_paths: List[Path],
        continue_on_error: bool = True,
    ) -> Tuple[List[ParsedDocument], List[Tuple[Path, Exception]]]:
        """
        Parse multiple files in batch.

        Args:
            file_paths: List of file paths to parse
            continue_on_error: If True, continue parsing even if some files fail

        Returns:
            Tuple of (successful_documents, failed_files_with_errors)
        """
        successful_docs = []
        failed_files = []

        logger.info(f"Batch parsing {len(file_paths)} files")

        for file_path in file_paths:
            try:
                doc = self.parse_file(file_path)
                successful_docs.append(doc)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                failed_files.append((file_path, e))

                if not continue_on_error:
                    raise

        logger.info(
            f"Batch parsing complete: {len(successful_docs)} successful, "
            f"{len(failed_files)} failed"
        )

        return successful_docs, failed_files

    def _extract_title(self, doc) -> str:
        """Extract document title from Docling document."""
        # Try to get title from metadata or first heading
        try:
            if hasattr(doc, "title") and doc.title:
                return doc.title

            # Look for first heading in content
            for item in doc.iterate_items():
                if hasattr(item, "label") and "TITLE" in str(item.label).upper():
                    return item.text if hasattr(item, "text") else ""
                if hasattr(item, "label") and "HEADING" in str(item.label).upper():
                    return item.text if hasattr(item, "text") else ""
        except Exception as e:
            logger.warning(f"Could not extract title: {e}")

        return ""

    def _extract_sections(self, doc) -> List[Dict]:
        """Extract section hierarchy from document."""
        sections = []

        try:
            for item in doc.iterate_items():
                if hasattr(item, "label") and "SECTION" in str(item.label).upper():
                    section = {
                        "text": item.text if hasattr(item, "text") else "",
                        "level": self._get_section_level(item),
                        "bbox": self._get_bbox(item),
                    }
                    sections.append(section)
        except Exception as e:
            logger.warning(f"Could not extract sections: {e}")

        return sections

    def _extract_equations(self, doc) -> List[Dict]:
        """Extract LaTeX equations from document."""
        equations = []

        try:
            for item in doc.iterate_items():
                if hasattr(item, "label") and "FORMULA" in str(item.label).upper():
                    equation = {
                        "latex": item.text if hasattr(item, "text") else "",
                        "bbox": self._get_bbox(item),
                        "type": "display" if "display" in str(item.label).lower() else "inline",
                    }

                    # Try to get enriched LaTeX if available
                    if hasattr(item, "latex") and item.latex:
                        equation["latex"] = item.latex

                    equations.append(equation)
        except Exception as e:
            logger.warning(f"Could not extract equations: {e}")

        return equations

    def _extract_figures(self, doc) -> List[Dict]:
        """Extract figure information and bounding boxes."""
        figures = []

        try:
            for item in doc.iterate_items():
                if hasattr(item, "label") and "PICTURE" in str(item.label).upper():
                    figure = {
                        "caption": self._get_caption(item),
                        "bbox": self._get_bbox(item),
                        "image_data": getattr(item, "image", None),
                    }
                    figures.append(figure)
        except Exception as e:
            logger.warning(f"Could not extract figures: {e}")

        return figures

    def _get_section_level(self, item) -> int:
        """Determine section nesting level."""
        # Try to infer from label or properties
        if hasattr(item, "level"):
            return int(item.level)

        # Default to level 1
        return 1

    def _get_bbox(self, item) -> Optional[Dict]:
        """Extract bounding box coordinates if available."""
        if hasattr(item, "prov") and item.prov:
            for prov in item.prov:
                if hasattr(prov, "bbox"):
                    return {
                        "x0": prov.bbox.l,
                        "y0": prov.bbox.t,
                        "x1": prov.bbox.r,
                        "y1": prov.bbox.b,
                        "page": getattr(prov, "page", None),
                    }
        return None

    def _get_caption(self, item) -> str:
        """Extract figure caption text."""
        if hasattr(item, "caption"):
            return str(item.caption)
        return ""


def parse_latex_file(
    file_path: Path,
    enable_formula_enrichment: bool = True,
    enable_figure_extraction: bool = True,
) -> ParsedDocument:
    """
    Convenience function to parse a single LaTeX/PDF file.

    Args:
        file_path: Path to the file to parse
        enable_formula_enrichment: Extract LaTeX from formulas
        enable_figure_extraction: Extract figure images

    Returns:
        ParsedDocument with extracted content and metadata
    """
    parser = DoclingParser(
        enable_formula_enrichment=enable_formula_enrichment,
        enable_figure_extraction=enable_figure_extraction,
    )
    return parser.parse_file(file_path)


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    # Configure logging
    logger.add("logs/docling_parser.log", rotation="10 MB")

    # Parse a sample file
    sample_file = Path("Documents/Aerospace_Structures_LaTeX/Ch01_4P.tex")

    if sample_file.exists():
        try:
            doc = parse_latex_file(sample_file)
            print(f"Successfully parsed: {doc.title}")
            print(f"Parsing time: {doc.parsing_time_seconds:.2f}s")
            print(f"Equations found: {len(doc.equations)}")
            print(f"Figures found: {len(doc.figures)}")
            print(f"Sections found: {len(doc.sections)}")

            # Show first 500 characters of markdown
            print("\nMarkdown preview:")
            print(doc.markdown_content[:500])
        except Exception as e:
            print(f"Parsing failed: {e}")
    else:
        print(f"Sample file not found: {sample_file}")
