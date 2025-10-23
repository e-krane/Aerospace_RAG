"""
Parser output validation for equation preservation and quality assurance.

Validates:
- Equation count preservation (95%+ target)
- LaTeX syntax validity
- Figure-caption associations
- Section hierarchy completeness
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import re

from loguru import logger


@dataclass
class ValidationResult:
    """Result of parser validation."""

    file_path: str
    passed: bool
    equation_preservation_rate: float
    latex_syntax_valid: bool
    figures_validated: int
    sections_complete: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ParserValidator:
    """
    Validate parser output for quality assurance.

    Target: 95%+ equation preservation rate
    """

    def __init__(self, min_equation_preservation: float = 0.95):
        """
        Initialize validator.

        Args:
            min_equation_preservation: Minimum equation preservation rate (default 95%)
        """
        self.min_preservation = min_equation_preservation

    def validate_parsed_document(
        self,
        original_file: Path,
        parsed_doc,
    ) -> ValidationResult:
        """
        Validate parser output against original document.

        Args:
            original_file: Path to original LaTeX file
            parsed_doc: Parsed document from docling_parser

        Returns:
            ValidationResult with pass/fail and detailed issues
        """
        issues = []
        warnings = []

        # Count equations in original
        original_equations = self._count_equations_in_latex(original_file)

        # Count equations in parsed output
        parsed_equations = self._count_equations_in_markdown(parsed_doc.markdown_content)

        # Calculate preservation rate
        if original_equations > 0:
            preservation_rate = parsed_equations / original_equations
        else:
            preservation_rate = 1.0

        if preservation_rate < self.min_preservation:
            issues.append(
                f"Equation preservation rate {preservation_rate:.1%} below target "
                f"{self.min_preservation:.1%}"
            )

        # Validate LaTeX syntax
        latex_valid = self._validate_latex_syntax(parsed_doc.markdown_content)
        if not latex_valid:
            issues.append("Invalid LaTeX syntax detected in equations")

        # Validate figures
        figures_ok = self._validate_figures(parsed_doc)
        if not figures_ok:
            warnings.append("Figure-caption associations incomplete")

        # Validate sections
        sections_ok = self._validate_sections(parsed_doc)
        if not sections_ok:
            warnings.append("Section hierarchy incomplete")

        # Determine pass/fail
        passed = (
            preservation_rate >= self.min_preservation
            and latex_valid
            and len(issues) == 0
        )

        return ValidationResult(
            file_path=str(original_file),
            passed=passed,
            equation_preservation_rate=preservation_rate,
            latex_syntax_valid=latex_valid,
            figures_validated=len(parsed_doc.figures) if hasattr(parsed_doc, 'figures') else 0,
            sections_complete=sections_ok,
            issues=issues,
            warnings=warnings,
        )

    def _count_equations_in_latex(self, latex_file: Path) -> int:
        """Count equations in original LaTeX file."""
        try:
            content = latex_file.read_text()

            # Count display equations
            display_patterns = [
                r'\\begin\{equation\}',
                r'\\begin\{align\}',
                r'\\begin\{eqnarray\}',
                r'\\begin\{gather\}',
                r'\\\[',
            ]

            count = 0
            for pattern in display_patterns:
                count += len(re.findall(pattern, content))

            logger.debug(f"Found {count} equations in {latex_file.name}")
            return count

        except Exception as e:
            logger.error(f"Failed to count equations in {latex_file}: {e}")
            return 0

    def _count_equations_in_markdown(self, markdown: str) -> int:
        """Count equations in parsed markdown."""
        # Count display equations
        display_count = len(re.findall(r'\$\$[^\$]+\$\$', markdown))

        # Count LaTeX blocks
        latex_blocks = len(re.findall(r'\\begin\{equation\}', markdown))
        latex_blocks += len(re.findall(r'\\begin\{align\}', markdown))

        total = display_count + latex_blocks
        logger.debug(f"Found {total} equations in parsed markdown")
        return total

    def _validate_latex_syntax(self, markdown: str) -> bool:
        """Validate LaTeX syntax in equations."""
        # Find all equations
        equations = re.findall(r'\$\$([^\$]+)\$\$', markdown)

        for eq in equations:
            # Check for balanced braces
            if eq.count('{') != eq.count('}'):
                logger.warning(f"Unbalanced braces in equation: {eq[:50]}...")
                return False

            # Check for balanced brackets
            if eq.count('[') != eq.count(']'):
                logger.warning(f"Unbalanced brackets in equation: {eq[:50]}...")
                return False

        return True

    def _validate_figures(self, parsed_doc) -> bool:
        """Validate figure-caption associations."""
        if not hasattr(parsed_doc, 'figures'):
            return True  # No figures to validate

        # Check that each figure has a caption
        for fig in parsed_doc.figures:
            if not fig.get('caption'):
                logger.warning(f"Figure without caption: {fig.get('id', 'unknown')}")
                return False

        return True

    def _validate_sections(self, parsed_doc) -> bool:
        """Validate section hierarchy completeness."""
        if not hasattr(parsed_doc, 'sections'):
            return True  # No sections to validate

        # Check that sections have text and level
        for section in parsed_doc.sections:
            if not section.get('text'):
                logger.warning("Section without text")
                return False
            if not section.get('level'):
                logger.warning("Section without level")
                return False

        return True


def validate_parser_on_corpus(
    corpus_dir: Path,
    parser,
    num_files: int = 10,
) -> Dict:
    """
    Validate parser on a corpus of documents.

    Args:
        corpus_dir: Directory containing LaTeX files
        parser: Parser instance (Docling or Marker)
        num_files: Number of files to validate

    Returns:
        Validation report dictionary
    """
    validator = ParserValidator()
    latex_files = list(corpus_dir.glob("*.tex"))[:num_files]

    results = []

    for latex_file in latex_files:
        logger.info(f"Validating {latex_file.name}...")

        try:
            # Parse document
            parsed_doc = parser.parse_document(latex_file)

            # Validate
            result = validator.validate_parsed_document(latex_file, parsed_doc)
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to validate {latex_file}: {e}")
            results.append(
                ValidationResult(
                    file_path=str(latex_file),
                    passed=False,
                    equation_preservation_rate=0.0,
                    latex_syntax_valid=False,
                    figures_validated=0,
                    sections_complete=False,
                    issues=[f"Parser failed: {e}"],
                )
            )

    # Generate report
    passed = sum(1 for r in results if r.passed)
    avg_preservation = sum(r.equation_preservation_rate for r in results) / len(results)

    report = {
        "total_files": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0,
        "avg_equation_preservation": avg_preservation,
        "results": results,
    }

    logger.info(
        f"Validation complete: {passed}/{len(results)} passed "
        f"({avg_preservation:.1%} avg equation preservation)"
    )

    return report


if __name__ == "__main__":
    logger.add("logs/validator.log", rotation="10 MB")

    # Example: Validate a single file
    validator = ParserValidator(min_equation_preservation=0.95)

    print("\nParser Validator Ready")
    print(f"Target: ≥95% equation preservation")
