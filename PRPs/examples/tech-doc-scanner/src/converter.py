"""
Document converter wrapper with error handling and statistics.

Provides high-level interface for document conversion with consistent error
handling, logging, and statistics collection.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

from .config import Config, DoclingConfig, OutputConfig, Pix2TextConfig
from .enrichment import Pix2TextFormulaEnrichmentModel
from .enrichment.base import EnrichmentStats
from .pipelines import PipelineFactory

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """
    Result of a document conversion.

    Attributes:
        input_path: Path to input PDF
        output_dir: Directory where outputs were saved
        success: Whether conversion succeeded
        elapsed_time: Conversion time in seconds
        stats: Formula recognition statistics (if applicable)
        error: Error message if conversion failed
        output_files: List of generated output files
    """

    input_path: Path
    output_dir: Path
    success: bool
    elapsed_time: float
    stats: Optional[EnrichmentStats] = None
    error: Optional[str] = None
    output_files: List[Path] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "success": self.success,
            "elapsed_time": self.elapsed_time,
            "stats": self.stats.to_dict() if self.stats else None,
            "error": self.error,
            "output_files": [str(f) for f in self.output_files],
        }


class DocumentConverterWrapper:
    """
    High-level wrapper for document conversion.

    Provides a simple interface for converting PDFs to various formats with
    consistent error handling, logging, and statistics collection.

    Example:
        >>> from tech_doc_scanner import DocumentConverterWrapper, Config
        >>>
        >>> config = Config()
        >>> converter = DocumentConverterWrapper(config=config)
        >>> result = converter.convert_file("input.pdf", "./output")
        >>>
        >>> if result.success:
        ...     print(f"Conversion took {result.elapsed_time:.1f}s")
        ...     if result.stats:
        ...         print(f"Formulas: {result.stats.total}")
    """

    def __init__(
        self,
        pipeline_type: str = "pix2text",
        docling_config: Optional[DoclingConfig] = None,
        pix2text_config: Optional[Pix2TextConfig] = None,
        output_config: Optional[OutputConfig] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize document converter.

        Args:
            pipeline_type: Type of pipeline ('standard', 'pix2text', 'vlm')
            docling_config: Docling configuration (or use config)
            pix2text_config: Pix2Text configuration (or use config)
            output_config: Output configuration (or use config)
            config: Complete configuration (overrides individual configs)

        Example:
            >>> # Using individual configs
            >>> converter = DocumentConverterWrapper(
            ...     pipeline_type='pix2text',
            ...     docling_config=DoclingConfig(),
            ...     pix2text_config=Pix2TextConfig(device='cuda')
            ... )
            >>>
            >>> # Or using complete config
            >>> config = Config.from_yaml('config.yaml')
            >>> converter = DocumentConverterWrapper(config=config)
        """
        # Use config if provided, otherwise use individual configs
        if config:
            self.docling_config = config.docling
            self.pix2text_config = config.pix2text
            self.output_config = config.output
        else:
            self.docling_config = docling_config or DoclingConfig()
            self.pix2text_config = pix2text_config or Pix2TextConfig()
            self.output_config = output_config or OutputConfig()

        self.pipeline_type = pipeline_type

        # Create pipeline
        logger.info(f"Initializing {pipeline_type} pipeline")
        pipeline = PipelineFactory.create_pipeline(
            pipeline_type, self.docling_config, self.pix2text_config
        )

        # Create document converter
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_cls=type(pipeline))}
        )

        # Store pipeline reference for stats access
        self.pipeline = pipeline

        logger.info("DocumentConverterWrapper initialized")

    def convert_file(
        self, input_path: str | Path, output_dir: Optional[str | Path] = None
    ) -> ConversionResult:
        """
        Convert a single PDF file.

        Args:
            input_path: Path to input PDF file
            output_dir: Output directory (default: from config)

        Returns:
            ConversionResult with conversion details

        Example:
            >>> converter = DocumentConverterWrapper()
            >>> result = converter.convert_file("document.pdf", "./output")
            >>> if result.success:
            ...     print(f"Success! Files: {result.output_files}")
            ... else:
            ...     print(f"Error: {result.error}")
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir) if output_dir else self.output_config.base_dir

        logger.info(f"Converting: {input_path}")
        start_time = time.time()

        try:
            # Validate input
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

            if not input_path.suffix.lower() == ".pdf":
                raise ValueError(f"Input must be PDF file: {input_path}")

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Convert document
            result = self.converter.convert(str(input_path))

            # Get statistics if available
            stats = self._get_stats()

            # Export outputs
            output_files = self._export_document(
                result.document, input_path.stem, output_dir, self.output_config
            )

            elapsed_time = time.time() - start_time

            logger.info(f"Conversion complete in {elapsed_time:.1f}s")

            return ConversionResult(
                input_path=input_path,
                output_dir=output_dir,
                success=True,
                elapsed_time=elapsed_time,
                stats=stats,
                output_files=output_files,
            )

        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Conversion failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            return ConversionResult(
                input_path=input_path,
                output_dir=output_dir,
                success=False,
                elapsed_time=elapsed_time,
                error=error_msg,
            )

    def convert_batch(
        self, input_paths: List[str | Path], output_dir: Optional[str | Path] = None
    ) -> List[ConversionResult]:
        """
        Convert multiple PDF files.

        Args:
            input_paths: List of paths to input PDF files
            output_dir: Output directory (default: from config)

        Returns:
            List of ConversionResult for each file

        Example:
            >>> converter = DocumentConverterWrapper()
            >>> results = converter.convert_batch(["doc1.pdf", "doc2.pdf"])
            >>> success_count = sum(1 for r in results if r.success)
            >>> print(f"Converted {success_count}/{len(results)} files")
        """
        results = []

        logger.info(f"Starting batch conversion of {len(input_paths)} files")

        for i, input_path in enumerate(input_paths, 1):
            logger.info(f"Processing file {i}/{len(input_paths)}: {input_path}")
            result = self.convert_file(input_path, output_dir)
            results.append(result)

        success_count = sum(1 for r in results if r.success)
        logger.info(
            f"Batch conversion complete: {success_count}/{len(input_paths)} successful"
        )

        return results

    def _get_stats(self) -> Optional[EnrichmentStats]:
        """Get statistics from pipeline if available."""
        try:
            # Access enrichment model from initialized pipelines
            for pipeline in self.converter.initialized_pipelines.values():
                if hasattr(pipeline, "enrichment_model"):
                    if isinstance(pipeline.enrichment_model, Pix2TextFormulaEnrichmentModel):
                        return pipeline.enrichment_model.get_stats()
        except Exception as e:
            logger.warning(f"Could not retrieve statistics: {e}")

        return None

    def _export_document(
        self, document, base_name: str, output_dir: Path, config: OutputConfig
    ) -> List[Path]:
        """Export document to configured formats."""
        output_files = []

        # Map image mode
        image_mode_map = {
            "embedded": ImageRefMode.EMBEDDED,
            "referenced": ImageRefMode.REFERENCED,
            "placeholder": ImageRefMode.PLACEHOLDER,
        }
        image_mode = image_mode_map.get(config.image_mode.value, ImageRefMode.REFERENCED)

        # Export each format
        for fmt in config.formats:
            try:
                if fmt == "markdown":
                    output_path = output_dir / f"{base_name}.md"
                    document.save_as_markdown(output_path, image_mode=image_mode)
                    output_files.append(output_path)
                    logger.info(f"Saved markdown: {output_path}")

                elif fmt == "html":
                    output_path = output_dir / f"{base_name}.html"
                    document.save_as_html(output_path)
                    output_files.append(output_path)
                    logger.info(f"Saved HTML: {output_path}")

                elif fmt == "json":
                    output_path = output_dir / f"{base_name}.json"
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(document.export_to_dict(), f, indent=2)
                    output_files.append(output_path)
                    logger.info(f"Saved JSON: {output_path}")

                elif fmt == "doctags":
                    output_path = output_dir / f"{base_name}.doctags"
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(document.export_to_doctags())
                    output_files.append(output_path)
                    logger.info(f"Saved doctags: {output_path}")

                else:
                    logger.warning(f"Unknown output format: {fmt}")

            except Exception as e:
                logger.error(f"Error exporting to {fmt}: {e}")

        return output_files
