"""
Marker parser fallback for Docling failures.

Performance target: 25 pages/second
Used when Docling parser fails or produces low-quality output.
"""

from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import subprocess
import tempfile
import json

from loguru import logger


@dataclass
class MarkerResult:
    """Result from Marker parser."""

    markdown_content: str
    metadata: Dict[str, Any]
    page_count: int
    processing_time: float
    figures: list = None
    equations: list = None


class MarkerParser:
    """
    Fallback parser using Marker for fast PDF processing.

    Target: 25 pages/second
    Use case: When Docling fails or for rapid batch processing
    """

    def __init__(
        self,
        output_format: str = "markdown",
        extract_images: bool = True,
    ):
        """
        Initialize Marker parser.

        Args:
            output_format: Output format (markdown, json)
            extract_images: Whether to extract images
        """
        self.output_format = output_format
        self.extract_images = extract_images

    def parse_document(self, file_path: Path) -> MarkerResult:
        """
        Parse document using Marker.

        Args:
            file_path: Path to PDF/LaTeX file

        Returns:
            MarkerResult with markdown content and metadata
        """
        import time

        start_time = time.time()

        try:
            # Convert to PDF if LaTeX
            if file_path.suffix == ".tex":
                pdf_path = self._latex_to_pdf(file_path)
            else:
                pdf_path = file_path

            # Run Marker
            markdown, metadata = self._run_marker(pdf_path)

            processing_time = time.time() - start_time

            # Extract metadata
            page_count = metadata.get("page_count", 0)
            figures = metadata.get("figures", [])
            equations = metadata.get("equations", [])

            logger.info(
                f"Marker parsed {file_path.name} in {processing_time:.2f}s "
                f"({page_count / processing_time:.1f} pages/sec)"
            )

            return MarkerResult(
                markdown_content=markdown,
                metadata=metadata,
                page_count=page_count,
                processing_time=processing_time,
                figures=figures,
                equations=equations,
            )

        except Exception as e:
            logger.error(f"Marker parsing failed for {file_path}: {e}")
            raise

    def _run_marker(self, pdf_path: Path) -> tuple[str, Dict]:
        """Run Marker CLI on PDF."""
        try:
            # Use marker-pdf CLI if installed
            # Otherwise fall back to Python API
            try:
                result = subprocess.run(
                    [
                        "marker_single",
                        str(pdf_path),
                        "--output_format",
                        self.output_format,
                        "--extract_images",
                        str(self.extract_images),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Parse output
                output = json.loads(result.stdout)
                markdown = output.get("markdown", "")
                metadata = output.get("metadata", {})

                return markdown, metadata

            except FileNotFoundError:
                # Fall back to Python API
                return self._run_marker_python_api(pdf_path)

        except Exception as e:
            logger.error(f"Marker execution failed: {e}")
            raise

    def _run_marker_python_api(self, pdf_path: Path) -> tuple[str, Dict]:
        """
        Run Marker using Python API.

        Falls back to this if CLI not available.
        """
        try:
            # Try importing marker
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models

            logger.info("Using Marker Python API")

            # Load models (cached after first call)
            models = load_all_models()

            # Convert PDF
            full_text, images, metadata = convert_single_pdf(
                str(pdf_path), models, output_format=self.output_format
            )

            return full_text, metadata

        except ImportError:
            logger.error("Marker not installed. Install with: pip install marker-pdf")
            raise
        except Exception as e:
            logger.error(f"Marker Python API failed: {e}")
            raise

    def _latex_to_pdf(self, latex_path: Path) -> Path:
        """
        Convert LaTeX to PDF for Marker processing.

        Args:
            latex_path: Path to .tex file

        Returns:
            Path to generated PDF
        """
        try:
            # Create temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)

                # Run pdflatex
                result = subprocess.run(
                    [
                        "pdflatex",
                        "-interaction=nonstopmode",
                        "-output-directory",
                        str(output_dir),
                        str(latex_path),
                    ],
                    capture_output=True,
                    text=True,
                )

                # Check for PDF
                pdf_path = output_dir / latex_path.with_suffix(".pdf").name

                if not pdf_path.exists():
                    raise RuntimeError(
                        f"pdflatex failed to generate PDF: {result.stderr}"
                    )

                # Copy to permanent location
                permanent_path = latex_path.with_suffix(".pdf")
                import shutil

                shutil.copy(pdf_path, permanent_path)

                logger.info(f"Converted {latex_path.name} to PDF")
                return permanent_path

        except Exception as e:
            logger.error(f"LaTeX to PDF conversion failed: {e}")
            raise


class ParserFallbackChain:
    """
    Intelligent parser fallback chain.

    1. Try Docling (high quality, slow)
    2. Fall back to Marker (fast, good quality)
    3. Report failure if both fail
    """

    def __init__(self, docling_parser, marker_parser: Optional[MarkerParser] = None):
        """
        Initialize fallback chain.

        Args:
            docling_parser: Primary Docling parser
            marker_parser: Fallback Marker parser
        """
        self.docling = docling_parser
        self.marker = marker_parser or MarkerParser()

    def parse_with_fallback(self, file_path: Path):
        """
        Parse document with intelligent fallback.

        Args:
            file_path: Path to document

        Returns:
            Parsed document from best available parser
        """
        # Try Docling first
        try:
            logger.info(f"Attempting Docling parse: {file_path.name}")
            result = self.docling.parse_document(file_path)

            # Validate quality
            if self._validate_docling_output(result):
                logger.info(f"Docling parse succeeded: {file_path.name}")
                return result, "docling"

            logger.warning(
                f"Docling output quality low for {file_path.name}, trying Marker"
            )

        except Exception as e:
            logger.warning(f"Docling failed for {file_path.name}: {e}")

        # Fall back to Marker
        try:
            logger.info(f"Attempting Marker parse: {file_path.name}")
            result = self.marker.parse_document(file_path)
            logger.info(f"Marker parse succeeded: {file_path.name}")
            return result, "marker"

        except Exception as e:
            logger.error(f"Marker also failed for {file_path.name}: {e}")
            raise RuntimeError(f"All parsers failed for {file_path.name}")

    def _validate_docling_output(self, result) -> bool:
        """
        Validate Docling output quality.

        Args:
            result: Docling parse result

        Returns:
            True if output is acceptable quality
        """
        # Check for minimum content
        if not hasattr(result, "markdown_content"):
            return False

        markdown = result.markdown_content
        if len(markdown.strip()) < 100:  # Too short
            return False

        # Check for equation preservation (basic heuristic)
        if "$$" not in markdown and "\\begin{equation}" not in markdown:
            # Might be fine if document has no equations
            # More sophisticated check would compare with source
            pass

        return True


if __name__ == "__main__":
    logger.add("logs/marker_parser.log", rotation="10 MB")

    # Example usage
    parser = MarkerParser()

    print("\nMarker Parser Ready")
    print("Target: 25 pages/second")
    print("Use as fallback when Docling fails")
