"""
End-to-End RAG Pipeline for Aerospace Documents.

Integrates all components:
- Document parsing (Docling/Marker)
- Semantic chunking with equation preservation
- Embedding generation (Qwen3-embedding:8b)
- Vector storage (Qdrant with binary quantization)
- Hybrid retrieval (BM25 + semantic)
- Reranking (ColBERT)
- Answer generation (Qwen3:latest)
- Citation tracking

Usage:
    # Indexing
    pipeline = RAGPipeline()
    pipeline.index_document("path/to/aerospace_textbook.pdf")

    # Querying
    response = pipeline.query("What is the Euler buckling formula?")
    print(response.answer)
    print(response.citations)
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import time

from loguru import logger

# Import existing components
from src.utils.config import get_config
from src.embeddings.ollama_qwen3_embedder import OllamaQwen3Embedder
from src.llm.client import LLMClient
from src.llm.citation_tracker import CitationTracker
from src.chunking.semantic_chunker import SemanticChunker
from src.storage.qdrant_client import AerospaceQdrantClient
from src.retrieval.two_stage_pipeline import TwoStageRetriever
from src.parsers.docling_parser import DoclingParser


@dataclass
class QueryResponse:
    """Response from RAG pipeline query."""

    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]

    # Performance metrics
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float

    # Metadata
    query: str
    model_used: str
    tokens_used: int

    # Quality indicators
    confidence: Optional[float] = None
    sources_count: int = 0


@dataclass
class IndexingResult:
    """Result from document indexing."""

    document_path: str
    chunks_created: int
    chunks_indexed: int

    # Performance
    parsing_time_ms: float
    chunking_time_ms: float
    embedding_time_ms: float
    indexing_time_ms: float
    total_time_ms: float

    # Metadata
    pages_processed: int
    equations_preserved: int
    figures_extracted: int

    success: bool = True
    errors: List[str] = field(default_factory=list)


class RAGPipeline:
    """
    Complete RAG pipeline for aerospace document Q&A.

    Features:
    - Parallel model loading (12GB VRAM)
    - Equation-aware chunking
    - Hybrid retrieval with reranking
    - Citation tracking
    - Performance monitoring

    Usage:
        # Initialize (loads models)
        pipeline = RAGPipeline()

        # Index documents
        result = pipeline.index_document("textbook.pdf")

        # Query
        response = pipeline.query("Explain beam bending")
        print(f"Answer: {response.answer}")
        print(f"Sources: {len(response.citations)}")
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        preload_models: bool = True,
    ):
        """
        Initialize RAG pipeline.

        Args:
            config_dir: Custom configuration directory
            preload_models: Load embedding and LLM models at initialization
        """
        # Load configuration
        self.config = get_config()

        # Initialize components (lazy loading if not preloading)
        self._embedder = None
        self._llm = None
        self._parser = None
        self._chunker = None
        self._storage = None
        self._retriever = None
        self._citation_tracker = None

        if preload_models:
            self._initialize_components()

        logger.info(
            f"RAG Pipeline initialized "
            f"(preload={preload_models}, vram_mode={self.config.models.vram.mode})"
        )

    def _initialize_components(self):
        """Initialize all pipeline components."""
        start = time.time()

        # Embedder (4.7GB VRAM)
        logger.info("Loading embedding model...")
        self._embedder = OllamaQwen3Embedder(
            model_name=self.config.models.embeddings.model,
            use_matryoshka=self.config.models.embeddings.matryoshka,
            reduced_dimensions=self.config.models.embeddings.reduced_dimensions,
            batch_size=self.config.models.embeddings.batch_size,
        )

        # LLM (5.2GB VRAM)
        logger.info("Loading LLM...")
        self._llm = LLMClient(
            provider=self.config.models.llm.provider,
            model=self.config.models.llm.model,
            temperature=self.config.models.llm.temperature,
            max_tokens=self.config.models.llm.max_tokens,
            max_retries=self.config.models.llm.max_retries,
        )

        # Parser
        self._parser = DoclingParser()

        # Chunker
        self._chunker = SemanticChunker(
            chunk_size=self.config.system.chunking.chunk_size,
            overlap=self.config.system.chunking.overlap,
        )

        # Storage
        self._storage = AerospaceQdrantClient(
            host=self.config.system.vector_db.host,
            port=self.config.system.vector_db.port,
        )

        # Retrieval
        self._retriever = TwoStageRetriever(
            qdrant_client=self._storage.client,
            collection_name=self.config.system.vector_db.collection_name,
        )

        # Citation tracker
        self._citation_tracker = CitationTracker()

        elapsed = (time.time() - start) * 1000
        logger.info(f"All components initialized in {elapsed:.0f}ms")

    def index_document(
        self,
        document_path: str,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> IndexingResult:
        """
        Index a document into the RAG system.

        Args:
            document_path: Path to PDF or document file
            batch_size: Embedding batch size
            show_progress: Show progress bars

        Returns:
            IndexingResult with metrics
        """
        total_start = time.time()
        doc_path = Path(document_path)

        if not doc_path.exists():
            return IndexingResult(
                document_path=str(doc_path),
                chunks_created=0,
                chunks_indexed=0,
                parsing_time_ms=0,
                chunking_time_ms=0,
                embedding_time_ms=0,
                indexing_time_ms=0,
                total_time_ms=0,
                pages_processed=0,
                equations_preserved=0,
                figures_extracted=0,
                success=False,
                errors=[f"File not found: {document_path}"],
            )

        logger.info(f"Indexing document: {doc_path.name}")

        # Ensure components are initialized
        if self._embedder is None:
            self._initialize_components()

        # 1. Parse document
        logger.info("Parsing document...")
        parse_start = time.time()
        parsed_doc = self._parser.parse(str(doc_path))
        parsing_time = (time.time() - parse_start) * 1000

        # 2. Chunk document
        logger.info("Chunking document...")
        chunk_start = time.time()
        chunks = self._chunker.chunk(parsed_doc["content"])
        chunking_time = (time.time() - chunk_start) * 1000

        # 3. Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embed_start = time.time()

        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self._embedder.embed(chunk_texts, show_progress=show_progress)

        embedding_time = (time.time() - embed_start) * 1000

        # 4. Store in Qdrant
        logger.info("Storing in vector database...")
        index_start = time.time()

        # Prepare points for Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append({
                "id": f"{doc_path.stem}_{i}",
                "vector": embedding.tolist(),
                "payload": {
                    "content": chunk.content,
                    "document": doc_path.name,
                    "chunk_index": i,
                    "metadata": getattr(chunk, "metadata", {}),
                },
            })

        # Batch upload to Qdrant
        self._storage.add_points(points)

        indexing_time = (time.time() - index_start) * 1000
        total_time = (time.time() - total_start) * 1000

        result = IndexingResult(
            document_path=str(doc_path),
            chunks_created=len(chunks),
            chunks_indexed=len(points),
            parsing_time_ms=parsing_time,
            chunking_time_ms=chunking_time,
            embedding_time_ms=embedding_time,
            indexing_time_ms=indexing_time,
            total_time_ms=total_time,
            pages_processed=parsed_doc.get("page_count", 0),
            equations_preserved=parsed_doc.get("equations_count", 0),
            figures_extracted=parsed_doc.get("figures_count", 0),
            success=True,
        )

        logger.info(
            f"✅ Indexing complete: {len(chunks)} chunks in {total_time:.0f}ms "
            f"({len(chunks) / (total_time / 1000):.1f} chunks/sec)"
        )

        return result

    def query(
        self,
        question: str,
        max_results: int = 5,
        include_sources: bool = True,
    ) -> QueryResponse:
        """
        Query the RAG system.

        Args:
            question: User question
            max_results: Maximum number of chunks to retrieve
            include_sources: Include source citations

        Returns:
            QueryResponse with answer and metadata
        """
        total_start = time.time()

        # Ensure components are initialized
        if self._llm is None:
            self._initialize_components()

        logger.info(f"Processing query: {question[:50]}...")

        # 1. Retrieve relevant chunks
        logger.debug("Retrieving relevant chunks...")
        retrieval_start = time.time()

        # Embed query
        query_embedding = self._embedder.embed([question])[0]

        # Retrieve and rerank
        retrieved = self._retriever.search(
            query_embedding=query_embedding.tolist(),
            query_text=question,
            top_k=max_results,
        )

        retrieval_time = (time.time() - retrieval_start) * 1000

        # 2. Generate answer
        logger.debug("Generating answer...")
        generation_start = time.time()

        # Extract context from retrieved chunks
        context = [chunk["payload"]["content"] for chunk in retrieved]

        # Generate answer with LLM
        llm_response = self._llm.generate(
            prompt=question,
            context=context,
        )

        generation_time = (time.time() - generation_start) * 1000

        # 3. Track citations
        citations = []
        if include_sources and self._citation_tracker:
            citations = self._citation_tracker.track(
                answer=llm_response.content,
                chunks=retrieved,
            )

        total_time = (time.time() - total_start) * 1000

        response = QueryResponse(
            answer=llm_response.content,
            citations=citations,
            retrieved_chunks=retrieved,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            query=question,
            model_used=llm_response.model,
            tokens_used=llm_response.tokens_used,
            sources_count=len(retrieved),
        )

        logger.info(
            f"✅ Query complete in {total_time:.0f}ms "
            f"(retrieval: {retrieval_time:.0f}ms, generation: {generation_time:.0f}ms)"
        )

        return response

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            "models_loaded": {
                "embedder": self._embedder is not None,
                "llm": self._llm is not None,
            },
            "config": {
                "embedding_model": self.config.models.embeddings.model,
                "llm_model": self.config.models.llm.model,
                "vram_mode": self.config.models.vram.mode,
            },
        }

        if self._storage:
            stats["vector_db"] = self._storage.get_collection_info()

        return stats


if __name__ == "__main__":
    # Example usage
    logger.add("logs/rag_pipeline.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print("RAG PIPELINE - End-to-End System")
    print("=" * 70)
    print("\nFeatures:")
    print("  • Qwen3-embedding:8b (4.7GB) - #1 MTEB embeddings")
    print("  • Qwen3:latest (5.2GB) - Answer generation")
    print("  • Hybrid retrieval (BM25 + semantic)")
    print("  • ColBERT reranking")
    print("  • Equation-aware chunking")
    print("  • Citation tracking")
    print("\nVRAM Usage (12GB GPU):")
    print("  • Parallel loading: 9.9GB / 12GB")
    print("  • Headroom: 2.1GB for processing")
    print("=" * 70 + "\n")

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = RAGPipeline(preload_models=True)

    print("\n✅ Pipeline ready!")
    print(f"\nStats: {pipeline.get_stats()}")
    print("\n" + "=" * 70 + "\n")
