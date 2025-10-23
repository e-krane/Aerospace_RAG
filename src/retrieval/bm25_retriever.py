"""
BM25 keyword search retriever using Qdrant sparse vectors.

Features:
- BM25 algorithm with configurable parameters (k1, b)
- Query preprocessing (lowercasing, optional stemming)
- Technical abbreviation and acronym handling
- Sparse vector generation for Qdrant
- Top-K retrieval with BM25 scores
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
from collections import defaultdict
import math

try:
    import numpy as np
except ImportError as e:
    raise ImportError("numpy required. Install with: pip install numpy") from e

from loguru import logger

# Import Qdrant client
try:
    from ..storage.qdrant_client import AerospaceQdrantClient
except ImportError:
    from storage.qdrant_client import AerospaceQdrantClient


@dataclass
class BM25Config:
    """BM25 algorithm configuration."""

    k1: float = 1.2  # Term frequency saturation
    b: float = 0.75  # Length normalization
    lowercase: bool = True
    remove_punctuation: bool = True
    use_stemming: bool = False


class BM25Retriever:
    """
    BM25 keyword search using Qdrant sparse vectors.

    Features:
    - Classic BM25 scoring (k1=1.2, b=0.75)
    - Query preprocessing for technical text
    - Abbreviation and acronym handling
    - Sparse vector generation for Qdrant
    """

    # Technical abbreviations common in aerospace/engineering
    TECHNICAL_ABBREVIATIONS = {
        "hnsw": "hierarchical navigable small world",
        "ann": "approximate nearest neighbor",
        "fem": "finite element method",
        "cfd": "computational fluid dynamics",
        "fem": "finite element method",
        "dof": "degrees of freedom",
        "fem": "finite element method",
        "ram": "random access memory",
    }

    def __init__(
        self,
        qdrant_client: AerospaceQdrantClient,
        config: Optional[BM25Config] = None,
    ):
        """
        Initialize BM25 retriever.

        Args:
            qdrant_client: Qdrant client instance
            config: BM25 configuration
        """
        self.client = qdrant_client
        self.config = config or BM25Config()

        logger.info(
            f"BM25 retriever initialized: k1={self.config.k1}, b={self.config.b}"
        )

    def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        query_filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search using BM25 keyword matching.

        Args:
            query: Query string
            collection_name: Qdrant collection name
            top_k: Number of results to return
            query_filter: Optional metadata filter

        Returns:
            List of results with BM25 scores
        """
        # Preprocess query
        processed_query = self._preprocess_query(query)

        # Generate sparse vector for query
        sparse_vector = self._text_to_sparse_vector(processed_query)

        # Search in Qdrant
        # Note: This is a simplified version - full BM25 in Qdrant requires
        # proper sparse vector indexing and scoring
        logger.info(f"BM25 search: '{query}' -> {len(sparse_vector['indices'])} terms")

        # For now, return empty results with note
        # In production, this would use Qdrant's sparse vector search
        logger.warning(
            "BM25 search requires Qdrant sparse vector configuration. "
            "This is a placeholder implementation."
        )

        return []

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query for BM25 search.

        Args:
            query: Raw query string

        Returns:
            Processed query string
        """
        text = query

        # Lowercase
        if self.config.lowercase:
            text = text.lower()

        # Expand abbreviations
        for abbrev, full_term in self.TECHNICAL_ABBREVIATIONS.items():
            # Match whole words only
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, full_term, text, flags=re.IGNORECASE)

        # Remove punctuation (except hyphens in technical terms)
        if self.config.remove_punctuation:
            # Keep hyphens, alphanumerics, and spaces
            text = re.sub(r'[^\w\s-]', ' ', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Split on whitespace
        tokens = text.split()

        # Remove empty tokens
        tokens = [t for t in tokens if t.strip()]

        return tokens

    def _text_to_sparse_vector(
        self,
        text: str,
    ) -> Dict:
        """
        Convert text to sparse vector for Qdrant.

        Args:
            text: Text to convert

        Returns:
            Dict with 'indices' and 'values' for sparse vector
        """
        # Tokenize
        tokens = self._tokenize(text)

        # Count term frequencies
        term_freq = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1

        # Create sparse vector
        # Map tokens to indices (simple hash for demo)
        indices = []
        values = []

        for term, freq in term_freq.items():
            # Use hash of term as index
            term_index = abs(hash(term)) % (2**32)
            indices.append(term_index)

            # BM25 term weight (simplified - missing IDF component)
            # Full BM25: (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
            # Simplified: just use term frequency for now
            weight = freq
            values.append(float(weight))

        return {
            "indices": indices,
            "values": values,
        }

    def compute_bm25_score(
        self,
        query_terms: List[str],
        document_text: str,
        doc_length: int,
        avg_doc_length: float,
        term_idfs: Dict[str, float],
    ) -> float:
        """
        Compute BM25 score for a document.

        Args:
            query_terms: Query terms
            document_text: Document text
            doc_length: Document length (word count)
            avg_doc_length: Average document length in collection
            term_idfs: IDF values for terms

        Returns:
            BM25 score
        """
        # Tokenize document
        doc_tokens = self._tokenize(document_text.lower())
        doc_term_freq = defaultdict(int)
        for token in doc_tokens:
            doc_term_freq[token] += 1

        # Compute BM25 score
        score = 0.0

        for term in query_terms:
            if term not in doc_term_freq:
                continue

            # Term frequency in document
            tf = doc_term_freq[term]

            # IDF for term
            idf = term_idfs.get(term, 0.0)

            # BM25 formula
            numerator = tf * (self.config.k1 + 1)
            denominator = tf + self.config.k1 * (
                1 - self.config.b + self.config.b * (doc_length / avg_doc_length)
            )

            term_score = idf * (numerator / denominator)
            score += term_score

        return score

    @staticmethod
    def compute_idf(
        term: str,
        num_docs: int,
        num_docs_with_term: int,
    ) -> float:
        """
        Compute IDF (Inverse Document Frequency) for a term.

        Args:
            term: Term to compute IDF for
            num_docs: Total number of documents
            num_docs_with_term: Number of documents containing the term

        Returns:
            IDF value
        """
        # IDF = log((N - n + 0.5) / (n + 0.5) + 1)
        # where N = total docs, n = docs with term
        return math.log(
            (num_docs - num_docs_with_term + 0.5) / (num_docs_with_term + 0.5) + 1
        )


def search_with_bm25(
    query: str,
    qdrant_client: AerospaceQdrantClient,
    collection_name: str,
    top_k: int = 10,
    k1: float = 1.2,
    b: float = 0.75,
) -> List[Dict]:
    """
    Convenience function for BM25 search.

    Args:
        query: Query string
        qdrant_client: Qdrant client
        collection_name: Collection name
        top_k: Number of results
        k1: BM25 k1 parameter
        b: BM25 b parameter

    Returns:
        Search results
    """
    config = BM25Config(k1=k1, b=b)
    retriever = BM25Retriever(qdrant_client, config=config)

    return retriever.search(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
    )


if __name__ == "__main__":
    # Example usage
    logger.add("logs/bm25_retriever.log", rotation="10 MB")

    # Test query preprocessing
    retriever = BM25Retriever(None)  # No client needed for preprocessing test

    test_queries = [
        "HNSW algorithm",
        "Euler buckling formula",
        "finite element analysis",
        "stress-strain relationship",
    ]

    print("\nBM25 Query Preprocessing:")
    for query in test_queries:
        processed = retriever._preprocess_query(query)
        sparse_vec = retriever._text_to_sparse_vector(processed)

        print(f"\nOriginal: {query}")
        print(f"Processed: {processed}")
        print(f"Tokens: {retriever._tokenize(processed)}")
        print(f"Sparse vector: {len(sparse_vec['indices'])} non-zero elements")

    # Test BM25 scoring
    print("\n\nBM25 Scoring Example:")

    query_terms = ["euler", "buckling"]
    doc_text = "The Euler buckling formula predicts column stability under compression."
    doc_length = len(doc_text.split())
    avg_doc_length = 50.0

    # Mock IDF values
    term_idfs = {
        "euler": 2.5,
        "buckling": 2.0,
        "formula": 1.5,
    }

    score = retriever.compute_bm25_score(
        query_terms=query_terms,
        document_text=doc_text,
        doc_length=doc_length,
        avg_doc_length=avg_doc_length,
        term_idfs=term_idfs,
    )

    print(f"Query: {' '.join(query_terms)}")
    print(f"Document: {doc_text}")
    print(f"BM25 Score: {score:.3f}")
