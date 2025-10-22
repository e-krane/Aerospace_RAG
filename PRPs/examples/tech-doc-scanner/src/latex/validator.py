"""
LaTeX validation module.

Validates LaTeX expressions for common errors before rendering.
"""

import re
from typing import Optional, Tuple


class LaTeXValidator:
    """Validate LaTeX expressions for common errors."""

    @staticmethod
    def validate(latex: str) -> Tuple[bool, Optional[str]]:
        """
        Validate LaTeX expression.

        Args:
            latex: LaTeX string to validate

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if validation passes
            - error_message: Description of error if validation fails, None otherwise

        Examples:
            >>> validator = LaTeXValidator()
            >>> validator.validate(r"\\frac{1}{2}")
            (True, None)
            >>> validator.validate(r"\\frac{1")
            (False, "Unbalanced braces: 1 open, 0 close")
        """
        if not latex or not isinstance(latex, str):
            return False, "Empty or invalid input"

        # Check 1: Balanced braces
        open_count = latex.count("{")
        close_count = latex.count("}")
        if open_count != close_count:
            return False, f"Unbalanced braces: {open_count} open, {close_count} close"

        # Check 2: Balanced environments
        for env in ["array", "matrix", "align", "equation"]:
            begin_count = latex.count(f"\\begin{{{env}}}")
            end_count = latex.count(f"\\end{{{env}}}")
            if begin_count != end_count:
                return (
                    False,
                    f"Unbalanced {env} environment: {begin_count} begin, {end_count} end",
                )

        # Check 3: Balanced \left and \right
        left_count = latex.count(r"\left")
        right_count = latex.count(r"\right")
        if left_count != right_count:
            return (
                False,
                f"Unbalanced \\left/\\right: {left_count} left, {right_count} right",
            )

        # Check 4: No incomplete commands at end
        if latex.rstrip().endswith(("_", "^", "\\")):
            return False, "Incomplete command at end"

        # Check 5: No obvious infinite patterns
        if re.search(r"(.)\1{100,}", latex):
            return False, "Excessive character repetition (possible infinite loop)"

        # Check 6: Reasonable length
        if len(latex) > 5000:
            return False, f"LaTeX too long ({len(latex)} chars, max 5000)"

        # Check 7: No double backslashes at end (incomplete)
        if latex.endswith("\\\\"):
            return False, "Incomplete line break at end"

        return True, None
