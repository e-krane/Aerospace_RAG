"""
Base classes for enrichment models.

Provides common functionality for custom enrichment models.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EnrichmentStats:
    """
    Statistics for enrichment processing.

    Attributes:
        total: Total elements processed
        recognized: Successfully recognized elements
        cleaned: Elements that required cleaning
        validation_failed: Elements that failed validation
        fallback_used: Elements that fell back to original
    """

    total: int = 0
    recognized: int = 0
    cleaned: int = 0
    validation_failed: int = 0
    fallback_used: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "recognized": self.recognized,
            "cleaned": self.cleaned,
            "validation_failed": self.validation_failed,
            "fallback_used": self.fallback_used,
        }

    @property
    def success_rate(self) -> float:
        """
        Calculate success rate.

        Returns:
            Success rate as percentage (0-100)
        """
        if self.total == 0:
            return 0.0
        return (self.recognized - self.fallback_used) / self.total * 100

    def print_summary(self) -> None:
        """Print statistics summary."""
        print("\n" + "=" * 70)
        print("Formula Recognition Statistics:")
        print(f"  Total formulas processed: {self.total}")
        print(f"  Successfully recognized: {self.recognized}")
        print(f"  Required cleaning: {self.cleaned}")
        print(f"  Validation failures: {self.validation_failed}")
        print(f"  Fallback to OCR: {self.fallback_used}")
        if self.total > 0:
            print(f"  Success rate: {self.success_rate:.1f}%")
        print("=" * 70 + "\n")
