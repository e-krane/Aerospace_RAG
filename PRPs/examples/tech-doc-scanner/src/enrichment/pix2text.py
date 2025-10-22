"""
Pix2Text enrichment model for formula recognition.

Uses Pix2Text's state-of-the-art MFR 1.5 model for mathematical formula recognition.
"""

import logging
from collections.abc import Iterable
from typing import Optional

from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling_core.types.doc import DocItemLabel, DoclingDocument, NodeItem, TextItem

from ..latex import LaTeXCleaner, LaTeXValidator
from .base import EnrichmentStats

# Import Pix2Text with availability check
try:
    from pix2text import Pix2Text

    PIX2TEXT_AVAILABLE = True
except ImportError:
    PIX2TEXT_AVAILABLE = False

logger = logging.getLogger(__name__)


class Pix2TextFormulaEnrichmentModel(BaseItemAndImageEnrichmentModel):
    """
    Custom enrichment model using Pix2Text for formula recognition.

    This model processes FORMULA elements detected by Docling and uses Pix2Text's
    MFR 1.5 model to recognize mathematical formulas. It includes:
    - LaTeX cleaning to fix common Pix2Text output errors
    - LaTeX validation to prevent rendering errors
    - Fallback to original OCR text when validation fails

    Attributes:
        images_scale: Scale factor for formula images (default: 2.6)
        expansion_factor: Context expansion around formulas (default: 0.1)
        enabled: Whether Pix2Text processing is enabled
        clean_output: Whether to clean LaTeX output
        validate: Whether to validate LaTeX before output
        fallback: Whether to fallback to OCR on validation failure
        device: Device for Pix2Text (cpu or cuda)
        stats: Processing statistics
    """

    images_scale: float = 2.6
    expansion_factor: float = 0.1

    def __init__(
        self,
        enabled: bool = True,
        clean_output: bool = True,
        validate: bool = True,
        fallback: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize Pix2Text enrichment model.

        Args:
            enabled: Enable Pix2Text formula recognition
            clean_output: Enable LaTeX cleaning
            validate: Enable LaTeX validation
            fallback: Fallback to OCR when validation fails
            device: Device for Pix2Text (cpu or cuda)

        Raises:
            ImportError: If pix2text is not installed when enabled=True
        """
        self.enabled = enabled
        self.clean_output = clean_output
        self.validate = validate
        self.fallback = fallback
        self.device = device
        self.stats = EnrichmentStats()

        # Initialize Pix2Text and helpers
        self.p2t: Optional[Pix2Text] = None
        self.cleaner: Optional[LaTeXCleaner] = None
        self.validator: Optional[LaTeXValidator] = None

        if self.enabled:
            if not PIX2TEXT_AVAILABLE:
                raise ImportError(
                    "pix2text is required but not installed. "
                    "Install with: pip install pix2text"
                )

            # Initialize Pix2Text model
            self.p2t = Pix2Text.from_config(device=self.device)
            logger.info(f"Pix2Text initialized on device: {self.device}")

            # Initialize LaTeX processing
            if self.clean_output:
                self.cleaner = LaTeXCleaner()
            if self.validate:
                self.validator = LaTeXValidator()

            logger.info(
                f"Pix2Text enrichment initialized (clean={clean_output}, "
                f"validate={validate}, fallback={fallback})"
            )

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        """
        Check if element is a formula that should be processed.

        Args:
            doc: Document being processed
            element: Element to check

        Returns:
            True if element should be processed, False otherwise
        """
        return (
            self.enabled
            and isinstance(element, TextItem)
            and element.label == DocItemLabel.FORMULA
        )

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        """
        Process batch of formula elements with Pix2Text.

        Args:
            doc: Document being processed
            element_batch: Batch of elements with images

        Yields:
            Processed NodeItem elements
        """
        if not self.enabled:
            return

        for enrich_element in element_batch:
            self.stats.total += 1
            original_text = enrich_element.item.text
            final_text = original_text  # Fallback to original

            try:
                # Get the cropped formula image
                formula_image = enrich_element.image

                # Use Pix2Text to recognize the formula
                result = self.p2t.recognize_formula(formula_image)

                if result and isinstance(result, str):
                    self.stats.recognized += 1

                    # Clean the LaTeX output
                    if self.clean_output and self.cleaner:
                        cleaned_result = self.cleaner.clean(result)

                        if cleaned_result != result:
                            self.stats.cleaned += 1
                            logger.debug(
                                f"LaTeX cleaned: '{result[:30]}...' -> '{cleaned_result[:30]}...'"
                            )
                        result = cleaned_result

                    # Validate the LaTeX
                    if self.validate and self.validator:
                        is_valid, error = self.validator.validate(result)

                        if is_valid:
                            final_text = result
                            logger.info(
                                f"✓ Formula recognized: '{original_text[:40]}...' -> "
                                f"'{result[:40]}...'"
                            )
                        else:
                            self.stats.validation_failed += 1
                            if self.fallback:
                                self.stats.fallback_used += 1
                                logger.warning(
                                    f"✗ Validation failed ({error}), using original OCR: "
                                    f"'{original_text[:40]}...'"
                                )
                                final_text = original_text
                            else:
                                # Use invalid LaTeX anyway
                                final_text = result
                                logger.warning(
                                    f"⚠ Validation failed ({error}) but fallback disabled: "
                                    f"'{result[:40]}...'"
                                )
                    else:
                        # No validation, use result
                        final_text = result
                        logger.info(
                            f"Formula recognized: '{original_text[:40]}...' -> "
                            f"'{result[:40]}...'"
                        )

                else:
                    logger.warning("Pix2Text returned empty result for formula")

            except Exception as e:
                logger.error(f"Error processing formula with Pix2Text: {e}")
                # Keep original text on error

            # Update the text content
            enrich_element.item.text = final_text
            yield enrich_element.item

    def print_stats(self) -> None:
        """Print processing statistics."""
        self.stats.print_summary()

    def get_stats(self) -> EnrichmentStats:
        """
        Get processing statistics.

        Returns:
            EnrichmentStats object with processing statistics
        """
        return self.stats
