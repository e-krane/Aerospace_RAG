"""
LaTeX cleaning module.

Cleans and fixes common LaTeX issues from Pix2Text output.
"""

import re


class LaTeXCleaner:
    """Clean and fix common LaTeX issues from Pix2Text output."""

    @staticmethod
    def balance_braces(latex: str) -> str:
        """
        Balance opening and closing braces.

        Args:
            latex: LaTeX string to fix

        Returns:
            LaTeX with balanced braces

        Examples:
            >>> cleaner = LaTeXCleaner()
            >>> cleaner.balance_braces(r"\\frac{1{2")
            '\\\\frac{1{2}}'
        """
        open_count = latex.count("{")
        close_count = latex.count("}")

        if open_count > close_count:
            latex += "}" * (open_count - close_count)
        elif close_count > open_count:
            extra = close_count - open_count
            for _ in range(extra):
                if latex.endswith("}"):
                    latex = latex[:-1]

        return latex

    @staticmethod
    def fix_array_environments(latex: str) -> str:
        """
        Fix malformed array/matrix environments.

        Args:
            latex: LaTeX string to fix

        Returns:
            LaTeX with balanced environments
        """
        for env in ["array", "matrix", "align", "equation"]:
            begin_count = latex.count(f"\\begin{{{env}}}")
            end_count = latex.count(f"\\end{{{env}}}")

            if begin_count > end_count:
                latex += f" \\end{{{env}}}" * (begin_count - end_count)

        return latex

    @staticmethod
    def fix_left_right(latex: str) -> str:
        """
        Balance \\left and \\right delimiters.

        Args:
            latex: LaTeX string to fix

        Returns:
            LaTeX with balanced left/right delimiters
        """
        left_count = latex.count(r"\left")
        right_count = latex.count(r"\right")

        if left_count > right_count:
            latex += r" \right." * (left_count - right_count)
        elif right_count > left_count:
            latex = r"\left. " * (right_count - left_count) + latex

        return latex

    @staticmethod
    def remove_infinite_patterns(latex: str) -> str:
        """
        Remove patterns that could cause infinite loops.

        Args:
            latex: LaTeX string to fix

        Returns:
            LaTeX with infinite patterns removed
        """
        # Remove recursive macro definitions
        latex = re.sub(r"\\def\\([a-zA-Z]+)\{\\1\}", "", latex)

        # Remove excessive repetitions
        latex = re.sub(r"(.)\1{50,}", r"\1\1\1", latex)

        return latex

    @staticmethod
    def clean_whitespace(latex: str) -> str:
        """
        Clean up excessive whitespace.

        Args:
            latex: LaTeX string to clean

        Returns:
            LaTeX with cleaned whitespace
        """
        latex = re.sub(r"  +", " ", latex)
        latex = re.sub(r"\{\s+", "{", latex)
        latex = re.sub(r"\s+\}", "}", latex)
        return latex.strip()

    @staticmethod
    def fix_common_errors(latex: str) -> str:
        """
        Fix common Pix2Text output errors.

        Args:
            latex: LaTeX string to fix

        Returns:
            LaTeX with common errors fixed
        """
        # Fix incomplete \frac at end
        latex = re.sub(r"\\frac\s*\{([^}]*)\}\s*$", r"\\frac{\1}{}", latex)

        # Fix incomplete subscripts/superscripts at end
        latex = re.sub(r"_\s*$", "", latex)
        latex = re.sub(r"\^\s*$", "", latex)

        # Remove trailing backslash
        if latex.endswith("\\") and not latex.endswith("\\\\"):
            latex = latex[:-1]

        # Fix double backslash at end (incomplete)
        latex = re.sub(r"\\\\\s*$", "", latex)

        return latex

    @classmethod
    def clean(cls, latex: str, max_iterations: int = 3) -> str:
        """
        Apply all cleaning steps iteratively.

        Args:
            latex: LaTeX string to clean
            max_iterations: Maximum number of cleaning iterations

        Returns:
            Cleaned LaTeX string

        Examples:
            >>> cleaner = LaTeXCleaner()
            >>> cleaner.clean(r"\\frac{1{2  ")
            '\\\\frac{1{2}}'
        """
        if not latex or not isinstance(latex, str):
            return latex

        for iteration in range(max_iterations):
            original = latex

            # Apply cleaning steps
            latex = cls.remove_infinite_patterns(latex)
            latex = cls.fix_array_environments(latex)
            latex = cls.fix_left_right(latex)
            latex = cls.fix_common_errors(latex)
            latex = cls.balance_braces(latex)
            latex = cls.clean_whitespace(latex)

            # Stop if no changes made
            if latex == original:
                break

        return latex
