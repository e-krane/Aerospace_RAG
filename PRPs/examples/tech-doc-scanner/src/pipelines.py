"""
Pipeline factory for creating configured Docling pipelines.

Provides factory methods to create different types of document processing pipelines
with appropriate configurations.
"""

import logging
from typing import Optional

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

from .config import DoclingConfig, Pix2TextConfig
from .enrichment import Pix2TextFormulaEnrichmentModel

logger = logging.getLogger(__name__)


class Pix2TextPipelineOptions(PdfPipelineOptions):
    """Extended pipeline options with Pix2Text formula recognition."""

    do_pix2text_formulas: bool = True
    clean_latex_output: bool = True
    validate_latex: bool = True
    fallback_on_error: bool = True


class Pix2TextPipeline(StandardPdfPipeline):
    """
    Extended PDF pipeline with Pix2Text formula recognition.

    This pipeline extends Docling's StandardPdfPipeline with Pix2Text formula
    recognition capabilities, including LaTeX cleaning and validation.
    """

    def __init__(self, pipeline_options: Pix2TextPipelineOptions, device: str = "cpu"):
        """
        Initialize Pix2Text pipeline.

        Args:
            pipeline_options: Pipeline configuration options
            device: Device for Pix2Text (cpu or cuda)
        """
        super().__init__(pipeline_options)
        self.pipeline_options: Pix2TextPipelineOptions = pipeline_options

        # Add Pix2Text enrichment to the pipeline
        self.enrichment_model = Pix2TextFormulaEnrichmentModel(
            enabled=self.pipeline_options.do_pix2text_formulas,
            clean_output=self.pipeline_options.clean_latex_output,
            validate=self.pipeline_options.validate_latex,
            fallback=self.pipeline_options.fallback_on_error,
            device=device,
        )
        self.enrichment_pipe = [self.enrichment_model]

        # Keep backend for image cropping
        if self.pipeline_options.do_pix2text_formulas:
            self.keep_backend = True

        logger.info(
            f"Pix2TextPipeline initialized (formulas={self.pipeline_options.do_pix2text_formulas}, "
            f"validate={self.pipeline_options.validate_latex}, device={device})"
        )

    @classmethod
    def get_default_options(cls) -> Pix2TextPipelineOptions:
        """Get default pipeline options."""
        return Pix2TextPipelineOptions()


class PipelineFactory:
    """
    Factory for creating configured document processing pipelines.

    Provides methods to create different types of pipelines with appropriate
    configurations based on DoclingConfig and Pix2TextConfig.
    """

    @staticmethod
    def create_standard_pipeline(config: DoclingConfig) -> StandardPdfPipeline:
        """
        Create standard Docling pipeline without formula recognition.

        Args:
            config: Docling configuration

        Returns:
            Configured StandardPdfPipeline

        Example:
            >>> config = DoclingConfig()
            >>> pipeline = PipelineFactory.create_standard_pipeline(config)
        """
        options = config.to_pipeline_options()
        logger.info("Creating standard PDF pipeline")
        return StandardPdfPipeline(options)

    @staticmethod
    def create_pix2text_pipeline(
        docling_config: DoclingConfig,
        pix2text_config: Pix2TextConfig,
    ) -> Pix2TextPipeline:
        """
        Create Pix2Text pipeline with formula recognition.

        This is the recommended pipeline for technical documents with equations.

        Args:
            docling_config: Docling configuration
            pix2text_config: Pix2Text configuration

        Returns:
            Configured Pix2TextPipeline

        Example:
            >>> docling_config = DoclingConfig()
            >>> pix2text_config = Pix2TextConfig(device='cuda')
            >>> pipeline = PipelineFactory.create_pix2text_pipeline(
            ...     docling_config, pix2text_config
            ... )
        """
        # Create base options from docling config
        base_options = docling_config.to_pipeline_options()

        # Create extended options with Pix2Text settings
        options = Pix2TextPipelineOptions()

        # Copy base options
        options.do_ocr = base_options.do_ocr
        options.ocr_options = base_options.ocr_options
        options.do_table_structure = base_options.do_table_structure
        options.table_structure_options = base_options.table_structure_options
        options.generate_picture_images = base_options.generate_picture_images
        options.images_scale = base_options.images_scale
        options.do_picture_classification = base_options.do_picture_classification
        options.do_code_enrichment = base_options.do_code_enrichment
        options.accelerator_options = base_options.accelerator_options

        # Add Pix2Text settings
        options.do_pix2text_formulas = pix2text_config.enabled
        options.clean_latex_output = pix2text_config.clean
        options.validate_latex = pix2text_config.validate
        options.fallback_on_error = pix2text_config.fallback_to_ocr

        logger.info(
            f"Creating Pix2Text pipeline (device={pix2text_config.device}, "
            f"validate={pix2text_config.validate}, fallback={pix2text_config.fallback_to_ocr})"
        )

        return Pix2TextPipeline(options, device=pix2text_config.device)

    @staticmethod
    def create_vlm_pipeline(config: DoclingConfig) -> StandardPdfPipeline:
        """
        Create VLM (Vision Language Model) pipeline.

        Note: This is a placeholder for future VLM pipeline implementation.
        Currently returns a standard pipeline.

        Args:
            config: Docling configuration

        Returns:
            Configured pipeline (currently StandardPdfPipeline)

        Example:
            >>> config = DoclingConfig()
            >>> pipeline = PipelineFactory.create_vlm_pipeline(config)
        """
        logger.warning("VLM pipeline not yet implemented, using standard pipeline")
        return PipelineFactory.create_standard_pipeline(config)

    @staticmethod
    def create_pipeline(
        pipeline_type: str,
        docling_config: DoclingConfig,
        pix2text_config: Optional[Pix2TextConfig] = None,
    ) -> StandardPdfPipeline:
        """
        Create pipeline by type string.

        Args:
            pipeline_type: Type of pipeline ('standard', 'pix2text', 'vlm')
            docling_config: Docling configuration
            pix2text_config: Pix2Text configuration (required for pix2text pipeline)

        Returns:
            Configured pipeline

        Raises:
            ValueError: If pipeline_type is invalid or pix2text_config is missing

        Example:
            >>> docling_config = DoclingConfig()
            >>> pix2text_config = Pix2TextConfig()
            >>> pipeline = PipelineFactory.create_pipeline(
            ...     'pix2text', docling_config, pix2text_config
            ... )
        """
        if pipeline_type == "standard":
            return PipelineFactory.create_standard_pipeline(docling_config)
        elif pipeline_type == "pix2text":
            if pix2text_config is None:
                raise ValueError("pix2text_config required for pix2text pipeline")
            return PipelineFactory.create_pix2text_pipeline(docling_config, pix2text_config)
        elif pipeline_type == "vlm":
            return PipelineFactory.create_vlm_pipeline(docling_config)
        else:
            raise ValueError(
                f"Invalid pipeline type: {pipeline_type}. "
                f"Must be 'standard', 'pix2text', or 'vlm'"
            )
