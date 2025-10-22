"""
Metadata enrichment for chunks to enable filtering and tracking.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import re
from loguru import logger


@dataclass
class EnrichedChunk:
    """Chunk with comprehensive metadata."""

    content: str
    chunk_id: str
    document_id: str
    section_path: List[str]
    chunk_type: str  # "text", "equation", "figure", "mixed"
    has_equations: bool
    equation_count: int
    figure_references: List[str]
    keywords: List[str]
    page_number: Optional[int]
    tokens: int
    latex_source: Optional[str]
    metadata: Dict = field(default_factory=dict)


class MetadataEnricher:
    """
    Add comprehensive metadata to chunks.

    Metadata includes:
    - document_id, section_path, chunk_id
    - chunk_type, has_equations, equation_count
    - figure_references, keywords
    - page_number, tokens, latex_source
    """

    def __init__(self, extract_keywords: bool = True):
        """
        Initialize metadata enricher.

        Args:
            extract_keywords: Whether to extract technical keywords
        """
        self.extract_keywords = extract_keywords

    def enrich_chunk(
        self,
        chunk,
        document_id: str,
        parsed_doc=None,
    ) -> EnrichedChunk:
        """
        Enrich a chunk with comprehensive metadata.

        Args:
            chunk: Base chunk object
            document_id: Document identifier
            parsed_doc: Optional parsed document for additional context

        Returns:
            EnrichedChunk with full metadata
        """
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)

        # Detect equations
        has_equations, equation_count = self._detect_equations(content)

        # Extract figure references
        figure_refs = self._extract_figure_references(content)

        # Extract keywords
        keywords = []
        if self.extract_keywords:
            keywords = self._extract_keywords(content)

        # Determine chunk type
        chunk_type = self._classify_chunk(content, has_equations, figure_refs)

        # Get section path
        section_path = []
        if hasattr(chunk, 'section_path'):
            section_path = chunk.section_path
        elif hasattr(chunk, 'metadata') and 'section_path' in chunk.metadata:
            section_path = chunk.metadata['section_path']

        # Create enriched chunk
        enriched = EnrichedChunk(
            content=content,
            chunk_id=chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"{document_id}_chunk",
            document_id=document_id,
            section_path=section_path,
            chunk_type=chunk_type,
            has_equations=has_equations,
            equation_count=equation_count,
            figure_references=figure_refs,
            keywords=keywords,
            page_number=self._get_page_number(chunk, parsed_doc),
            tokens=chunk.token_count if hasattr(chunk, 'token_count') else len(content.split()),
            latex_source=self._extract_latex(content) if has_equations else None,
            metadata={
                **(chunk.metadata if hasattr(chunk, 'metadata') else {}),
                'enriched': True,
            }
        )

        return enriched

    def enrich_chunks(
        self,
        chunks: List,
        document_id: str,
        parsed_doc=None,
    ) -> List[EnrichedChunk]:
        """
        Enrich multiple chunks.

        Args:
            chunks: List of chunks
            document_id: Document identifier
            parsed_doc: Optional parsed document

        Returns:
            List of enriched chunks
        """
        enriched_chunks = []

        for chunk in chunks:
            try:
                enriched = self.enrich_chunk(chunk, document_id, parsed_doc)
                enriched_chunks.append(enriched)
            except Exception as e:
                logger.warning(f"Failed to enrich chunk: {e}")

        logger.info(
            f"Enriched {len(enriched_chunks)} chunks with metadata "
            f"({sum(c.has_equations for c in enriched_chunks)} with equations)"
        )

        return enriched_chunks

    def _detect_equations(self, text: str) -> tuple[bool, int]:
        """Detect LaTeX equations in text."""
        # Display equations
        display_patterns = [
            r'\$\$[^\$]+\$\$',
            r'\\begin\{equation\}',
            r'\\begin\{align\}',
            r'\\\[',
        ]

        equation_count = 0
        for pattern in display_patterns:
            equation_count += len(re.findall(pattern, text))

        # Inline equations
        inline_count = len(re.findall(r'\$[^\$]+\$', text))
        equation_count += inline_count

        return equation_count > 0, equation_count

    def _extract_figure_references(self, text: str) -> List[str]:
        """Extract figure references like Fig. 1, Figure 2-1, etc."""
        patterns = [
            r'Fig\.?\s*(\d+(?:-\d+)?)',
            r'Figure\s+(\d+(?:-\d+)?)',
            r'fig\.?\s*(\d+(?:-\d+)?)',
        ]

        refs = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            refs.extend(matches)

        return list(set(refs))  # Remove duplicates

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract technical keywords (simple implementation).

        For production, use spaCy or KeyBERT for better results.
        """
        # Simple approach: extract capitalized technical terms
        words = text.split()
        keywords = []

        # Look for capitalized words (potential technical terms)
        for word in words:
            cleaned = re.sub(r'[^\w]', '', word)
            if cleaned and cleaned[0].isupper() and len(cleaned) > 3:
                if not cleaned.isupper():  # Exclude all-caps (likely acronyms)
                    keywords.append(cleaned)

        # Extract common technical patterns
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+(?:ness|tion|ment|ity)\b',  # Technical suffixes
        ]

        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)

        return list(set(keywords))[:10]  # Top 10 unique keywords

    def _classify_chunk(
        self,
        text: str,
        has_equations: bool,
        figure_refs: List[str],
    ) -> str:
        """Classify chunk type."""
        if has_equations and figure_refs:
            return "mixed"
        elif has_equations:
            return "equation"
        elif figure_refs:
            return "figure"
        else:
            return "text"

    def _get_page_number(self, chunk, parsed_doc) -> Optional[int]:
        """Extract page number if available."""
        if hasattr(chunk, 'metadata') and 'page_number' in chunk.metadata:
            return chunk.metadata['page_number']
        return None

    def _extract_latex(self, text: str) -> Optional[str]:
        """Extract LaTeX source from text."""
        # Find first equation
        match = re.search(r'\$\$([^\$]+)\$\$', text)
        if match:
            return match.group(1)

        match = re.search(r'\$([^\$]+)\$', text)
        if match:
            return match.group(1)

        return None


def enrich_chunks_with_metadata(
    chunks: List,
    document_id: str,
    parsed_doc=None,
) -> List[EnrichedChunk]:
    """
    Convenience function to enrich chunks.

    Args:
        chunks: List of chunks to enrich
        document_id: Document identifier
        parsed_doc: Optional parsed document

    Returns:
        List of enriched chunks
    """
    enricher = MetadataEnricher(extract_keywords=True)
    return enricher.enrich_chunks(chunks, document_id, parsed_doc)


if __name__ == "__main__":
    # Example usage
    from dataclasses import dataclass

    @dataclass
    class SimpleChunk:
        content: str
        chunk_id: str
        token_count: int
        metadata: dict

    sample_chunk = SimpleChunk(
        content="The Euler buckling formula is $$P_{cr} = \\frac{\\pi^2 EI}{L^2}$$. See Figure 2-1.",
        chunk_id="test_001",
        token_count=20,
        metadata={}
    )

    enricher = MetadataEnricher()
    enriched = enricher.enrich_chunk(sample_chunk, "test_doc")

    print(f"Chunk type: {enriched.chunk_type}")
    print(f"Has equations: {enriched.has_equations}")
    print(f"Equation count: {enriched.equation_count}")
    print(f"Figure refs: {enriched.figure_references}")
    print(f"Keywords: {enriched.keywords}")
