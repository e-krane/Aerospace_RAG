## Revised LaTeX RAG System Architecture

### Core Principle
**"Optimal LaTeX document retrieval with equation and figure awareness"**

## Version 1: Core Functionality

### Storage Strategy (LaTeX + Markdown)
```
Original LaTeX (.tex) → Parse & Annotate → Hybrid Storage
                              ↓
                     [LaTeX source chunks]  // Stored as-is for equations
                              +
                     [Markdown metadata]    // For searchable annotations
                              ↓
                        Vector Database
```

**Implementation:**
- Store original LaTeX chunks with preserved equation environments
- Generate markdown annotations for each chunk (summaries, keywords) 
- Each database entry contains:
  ```json
  {
    "id": "chunk_123",
    "latex_content": "\\begin{equation}...",  // Original LaTeX
    "markdown_annotation": "This section derives...",  // LLM summary
    "figure_refs": ["fig_2.1", "fig_2.2"],
    "source_location": {"file": "chapter2.tex", "lines": [45, 89]}
  }
  ```

### Processing Pipeline

**1. LaTeX Parsing & Chunking**
- Parse `.tex` files directly using plasTeX/pylatexenc
- Simple section-based chunking with overlap
- Preserve all LaTeX commands, environments, and equations as-is
- Extract figure references and captions

**2. Annotation Layer**
- LLM generates markdown descriptions of each LaTeX chunk
- Extract semantic keywords and concepts
- Identify equation dependencies and variable definitions
- Create bidirectional figure-text mappings

**3. Dual Indexing**
- **Semantic Index**: Markdown annotations → embeddings → vector search
- **Exact Match Index**: Raw LaTeX for equation/command searches

### Retrieval Architecture

```
Query → Parallel Search → Fusion → Results with Context
         ├─ Semantic (on markdown annotations)
         └─ Keyword (on LaTeX source)
                ↓
         Retrieve linked figures
         Include equation context
```

### Figure Processing (Simplified)
- Extract figures with basic metadata (caption, label, source location)
- Use standard Tesseract OCR only if needed (most content is in LaTeX already)
- Generate text descriptions via LLM for complex diagrams
- Create bidirectional links without complex visual embeddings

## Version 2: Future Enhancements

### Advanced Features (Priority Order)
1. **Equation dependency graph** - Track which equations build on others
2. **Multi-file reference resolution** - Handle `\ref{}` across chapters
3. **Proof/theorem awareness** - Keep mathematical arguments together
4. **CLIP visual embeddings** - For figure similarity search
5. **Citation network** - Build paper reference graph
6. **Incremental updates** - Process only changed sections
7. **Custom LaTeX macro expansion** - Handle document-specific commands
8. **Compilation-free preview** - Generate approximate rendered views

### Retrieval Enhancements
- Cross-encoder reranking for better precision
- Query expansion using equation symbols
- Hierarchical retrieval (section → subsection → content)
- Multi-modal fusion scoring

### System Optimizations
- Caching layer for frequent queries
- Async processing pipeline
- Batch embedding generation
- Dynamic chunk sizing based on content type

## Technical Stack (V1)

### Essential Components
- **Parser**: plasTeX (handles complex LaTeX better than pandoc)
- **Vector DB**: ChromaDB (simple, sufficient for non-commercial use)
- **Embeddings**: OpenAI text-embedding-3-large or all-mpnet-base-v2 (free alternative)
- **LLM Annotation**: GPT-4 or Claude (for quality annotations)
- **Search**: ChromaDB similarity + simple regex for LaTeX commands
- **OCR**: Tesseract (only when absolutely needed)

### Why This Approach Works for LaTeX RAG

1. **Preserves LaTeX fidelity** - No information loss from conversion
2. **Searchable annotations** - Markdown layer enables semantic search
3. **Equation-aware** - Can find exact equations or similar mathematical concepts
4. **Simple but complete** - V1 handles core use case without over-engineering

### Implementation Priority

**First:** Get basic LaTeX parsing and chunking working
**Second:** Add LLM annotations and build dual index
**Third:** Implement bidirectional figure-text links
**Fourth:** Test retrieval quality and iterate

This approach gives you the best of both worlds: exact LaTeX preservation for technical accuracy and semantic search capabilities through the annotation layer.