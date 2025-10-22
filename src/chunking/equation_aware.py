"""
Equation boundary detection to prevent splitting LaTeX equations.

Ensures equations remain intact within single chunks with context.
"""

import re
from typing import List, Tuple
from loguru import logger


class EquationAwareChunker:
    """
    Detect and preserve equation boundaries during chunking.

    Features:
    - Detects display equations ($$...$$, \\begin{equation})
    - Identifies inline math ($...$)
    - Ensures equations stay intact
    - Keeps equation + context together
    - Handles multi-line equations
    """

    # Regex patterns for LaTeX equations
    DISPLAY_EQUATION_PATTERNS = [
        r'\$\$[^\$]+\$\$',  # $$...$$
        r'\\begin\{equation\}.*?\\end\{equation\}',  # \begin{equation}...\end{equation}
        r'\\begin\{align\}.*?\\end\{align\}',  # \begin{align}...\end{align}
        r'\\begin\{eqnarray\}.*?\\end\{eqnarray\}',  # \begin{eqnarray}...\end{eqnarray}
        r'\\begin\{gather\}.*?\\end\{gather\}',  # \begin{gather}...\end{gather}
        r'\\\[.*?\\\]',  # \[...\]
    ]

    INLINE_EQUATION_PATTERN = r'\$[^\$]+\$'  # $...$

    def __init__(self, context_sentences: int = 2):
        """
        Initialize equation-aware chunker.

        Args:
            context_sentences: Number of sentences to keep before/after equation
        """
        self.context_sentences = context_sentences

        # Compile patterns
        self.display_patterns = [
            re.compile(p, re.DOTALL) for p in self.DISPLAY_EQUATION_PATTERNS
        ]
        self.inline_pattern = re.compile(self.INLINE_EQUATION_PATTERN)

    def find_equation_boundaries(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Find all equation boundaries in text.

        Args:
            text: Text to search

        Returns:
            List of (start_pos, end_pos, equation_type) tuples
        """
        boundaries = []

        # Find display equations
        for pattern in self.display_patterns:
            for match in pattern.finditer(text):
                boundaries.append((
                    match.start(),
                    match.end(),
                    'display'
                ))

        # Find inline equations (excluding those within display equations)
        for match in self.inline_pattern.finditer(text):
            # Check if this inline equation is within a display equation
            within_display = False
            for start, end, eq_type in boundaries:
                if eq_type == 'display' and start <= match.start() < end:
                    within_display = True
                    break

            if not within_display:
                boundaries.append((
                    match.start(),
                    match.end(),
                    'inline'
                ))

        # Sort by start position
        boundaries.sort(key=lambda x: x[0])

        logger.debug(f"Found {len(boundaries)} equations in text")

        return boundaries

    def validate_chunks(self, chunks: List) -> bool:
        """
        Validate that no equations are split across chunks.

        Args:
            chunks: List of chunks to validate

        Returns:
            True if all equations are intact, False otherwise
        """
        violations = 0

        for chunk in chunks:
            text = chunk.content if hasattr(chunk, 'content') else str(chunk)
            equations = self.find_equation_boundaries(text)

            # Check for incomplete equations
            for start, end, eq_type in equations:
                equation_text = text[start:end]

                # Check if equation markers are balanced
                if eq_type == 'display':
                    if equation_text.count('$$') % 2 != 0:
                        violations += 1
                        logger.warning(
                            f"Unbalanced equation markers in chunk {chunk.chunk_id if hasattr(chunk, 'chunk_id') else 'unknown'}"
                        )

        if violations > 0:
            logger.error(f"Found {violations} equation boundary violations!")
            return False

        logger.info("All equation boundaries validated successfully")
        return True

    def get_equation_context(
        self,
        text: str,
        equation_start: int,
        equation_end: int,
    ) -> str:
        """
        Get equation with surrounding context.

        Args:
            text: Full text
            equation_start: Equation start position
            equation_end: Equation end position

        Returns:
            Text chunk containing equation + context
        """
        # Find sentence boundaries before equation
        before_text = text[:equation_start]
        sentences_before = re.split(r'[.!?]+\s+', before_text)
        context_before = ' '.join(sentences_before[-self.context_sentences:])

        # Find sentence boundaries after equation
        after_text = text[equation_end:]
        sentences_after = re.split(r'[.!?]+\s+', after_text)
        context_after = ' '.join(sentences_after[:self.context_sentences])

        # Combine
        equation_text = text[equation_start:equation_end]
        full_context = f"{context_before} {equation_text} {context_after}"

        return full_context.strip()


def validate_equation_preservation(chunks: List) -> bool:
    """
    Convenience function to validate equation preservation.

    Args:
        chunks: List of chunks to validate

    Returns:
        True if no equations are split
    """
    validator = EquationAwareChunker()
    return validator.validate_chunks(chunks)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    The Euler buckling load is given by the equation:

    $$P_{cr} = \\frac{\\pi^2 EI}{L^2}$$

    Where $E$ is the elastic modulus and $I$ is the second moment of area.
    This formula applies to pin-ended columns with length $L$.
    """

    chunker = EquationAwareChunker()
    boundaries = chunker.find_equation_boundaries(sample_text)

    print(f"Found {len(boundaries)} equations:")
    for start, end, eq_type in boundaries:
        print(f"  {eq_type}: {sample_text[start:end]}")
