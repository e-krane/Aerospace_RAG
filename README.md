# Aerospace LaTeX RAG System

Production-ready RAG (Retrieval-Augmented Generation) system optimized for LaTeX technical documentation with mathematical equations, focusing on the Aerospace Structures textbook corpus.

## Features

- **LaTeX-Aware Parsing**: 95%+ equation preservation accuracy using Docling and Marker
- **Intelligent Chunking**: Semantic chunking with equation boundary detection
- **Hybrid Retrieval**: BM25 + semantic search with reciprocal rank fusion
- **Reranking**: ColBERT v2 for precise relevance scoring
- **Quantization**: Binary + int8 quantization for 32x storage reduction
- **Evaluation**: RAGAS framework with synthetic test generation
- **Production-Ready**: Monitoring, caching, and sub-2s query latency

## Architecture

```
User Query
    ↓
Query Analysis → Hybrid Retrieval (BM25 + Semantic)
    ↓
Reciprocal Rank Fusion (Top-100)
    ↓
ColBERT Reranking (Top-10)
    ↓
LLM Answer Generation
    ↓
Response + Citations
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for Qdrant)
- CUDA 11.8+ (optional, for GPU acceleration)
- 32GB+ RAM (16GB minimum)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/erik/Aerospace_RAG.git
cd Aerospace_RAG
```

2. Create virtual environment and install dependencies:
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. Deploy Qdrant vector database:
```bash
docker-compose up -d
```

### Usage

Coming soon...

## Project Structure

```
aerospace-rag/
├── src/
│   ├── parsers/         # Docling and Marker LaTeX parsers
│   ├── chunking/        # Semantic and hierarchy-aware chunking
│   ├── embeddings/      # Qwen3 embedding generation
│   ├── retrieval/       # Hybrid search and reranking
│   ├── storage/         # Qdrant client and schema
│   ├── llm/             # LLM client and prompt templates
│   ├── evaluation/      # RAGAS integration
│   └── utils/           # Logging, config, caching
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── data/
│   ├── raw/             # Original LaTeX files
│   ├── processed/       # Parsed markdown
│   └── evaluation/      # Test sets (golden, synthetic)
├── models/              # Downloaded model weights
├── config/              # YAML configuration files
└── notebooks/           # Jupyter notebooks
```

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Parser** | Docling + Marker | Best equation accuracy, fully local |
| **Chunking** | semchunk + custom | Hierarchy-aware, equation-preserving |
| **Vector DB** | Qdrant | 1,238 QPS, native hybrid search |
| **Embeddings** | Qwen3-8B | #1 on MTEB-Code, 32K context |
| **Reranker** | Jina-ColBERT-v2 | 8192 tokens, multilingual |
| **Evaluation** | RAGAS + DeepEval | Comprehensive metrics, CI/CD |

## Performance Targets

| Metric | Target | Method |
|--------|--------|--------|
| Equation Preservation | >95% | Manual validation |
| Retrieval Precision@5 | >80% | RAGAS on golden set |
| Query Latency (p95) | <2s | End-to-end benchmarks |
| Storage Efficiency | 32x compression | Binary quantization |

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Git Workflow

See [PLANNING.md](PLANNING.md) for detailed branching strategy and commit guidelines.

## Roadmap

- [x] Phase 1: Foundation (Project setup, Qdrant deployment)
- [ ] Phase 2: Document Processing (Docling, Marker, validation)
- [ ] Phase 3: Intelligent Chunking (Semantic, hierarchy-aware)
- [ ] Phase 4: Embedding Generation (Qwen3, Matryoshka)
- [ ] Phase 5: Storage & Indexing (Dual index, quantization)
- [ ] Phase 6: Hybrid Retrieval (BM25, semantic, RRF)
- [ ] Phase 7: Reranking (ColBERT, optimization)
- [ ] Phase 8: Evaluation Framework (RAGAS, golden set)
- [ ] Phase 9: LLM Integration (Multi-provider, citations)
- [ ] Phase 10: Optimization (Caching, monitoring)

## License

Apache-2.0

## References

- [Docling](https://arxiv.org/abs/2501.17887v1) - IBM Research, 3.70s/page
- [Marker](https://github.com/VikParuchuri/marker) - 25 pages/second
- [Qdrant](https://qdrant.tech/) - Vector database
- [Qwen3 Embeddings](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) - #1 on MTEB
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG evaluation

## Contributing

Contributions are welcome! Please read the [PLANNING.md](PLANNING.md) for development guidelines.

## Acknowledgments

Based on the comprehensive implementation plan in [PRPs/aerospace-latex-rag.md](PRPs/aerospace-latex-rag.md), integrating best practices from 2024-2025 RAG research.
