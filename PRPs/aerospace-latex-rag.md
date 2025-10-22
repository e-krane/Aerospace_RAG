# Implementation Plan: Aerospace LaTeX RAG System

## Overview
Build a production-ready RAG system optimized for LaTeX technical documentation with mathematical equations, focusing on the Aerospace Structures textbook corpus. The system will achieve 95%+ equation accuracy, sub-second query response, and efficient storage through modern 2024-2025 techniques.

## Requirements Summary
- Parse LaTeX documents preserving mathematical notation with 95%+ accuracy
- Implement hybrid search (semantic + keyword) for technical terminology
- Achieve sub-second query latency for complex technical questions
- Support multimodal retrieval (text, equations, figures)
- Maintain document hierarchy and cross-references
- Provide evaluation framework for continuous quality monitoring
- Fully open-source stack suitable for non-commercial use
- **Maintain clean Git history with regular commits, feature branches, and proper issue tracking**

## Research Findings

### Best Practices from 2024-2025 Research
- **Simplification Movement**: Avoid complex abstractions like traditional LangChain; use lightweight frameworks (LightRAG, DSPy)
- **Semantic Chunking**: 80th percentile embedding similarity thresholds provide optimal relevancy; hierarchy-aware processing improves scores from 69.2% to 84.0%
- **Hybrid Search Essential**: 15-30% improvement in retrieval quality for technical documentation
- **Two-Stage Retrieval**: Broad initial retrieval (top-100) followed by precise reranking (to top-10) using ColBERT
- **Matryoshka Embeddings**: 768→256 dimensions retains 99.5% performance with 3x compression
- **Binary Quantization**: 32x compression with 96% performance when combined with int8 rescoring

### Reference Implementations
- **Docling** (IBM Research): arXiv:2501.17887v1 - 3.70s/page, 60.6 pages/second throughput, excellent LaTeX equation handling
- **Marker** (VikParuchuri): 25 pages/second, 95.67 heuristic score, texify model for equations
- **Qdrant**: 1,238 QPS at 99% recall, 3.5ms average latency, native hybrid search
- **Qwen3-8B Embeddings**: #1 on MTEB-Code and multilingual benchmarks, 32K context, Apache 2.0 license
- **Jina-ColBERT-v2**: 89 languages, 8192 token support for reranking
- **RAGAS**: 5.3k+ GitHub stars, synthetic test generation reducing manual effort by 90%

### Technology Decisions

| Component | Decision | Rationale |
|-----------|----------|-----------|
| **Parser** | Docling (primary), Marker (fallback) | Best equation accuracy, fully local, LangChain integration |
| **Chunking** | semchunk library + hierarchy-aware | 85% faster than alternatives, respects document structure |
| **Vector DB** | Qdrant | Top performance (1,238 QPS), native hybrid search, 24x compression |
| **Embeddings** | Qwen3-8B with Matryoshka | #1 for code+math, 100+ languages, open source |
| **Reranker** | Jina-ColBERT-v2 | 8192 tokens, multilingual, late-interaction accuracy |
| **Framework** | LightRAG | Simplicity, 14.6k stars, consistent benchmark wins |
| **Evaluation** | RAGAS + DeepEval | Comprehensive metrics, synthetic generation, CI/CD integration |

## Git Workflow and Issue Tracking

### Branching Strategy

**Main Branches:**
- `main` - Production-ready code, always stable
- `develop` - Integration branch for features, staging for next release

**Feature Branches:**
- `feature/parser-docling` - Docling parser implementation (Phase 2.1-2.3)
- `feature/chunking-semantic` - Semantic chunking (Phase 3.1-3.4)
- `feature/embeddings-qwen3` - Qwen3 embedding pipeline (Phase 4.1-4.4)
- `feature/storage-qdrant` - Qdrant integration (Phase 5.1-5.4)
- `feature/retrieval-hybrid` - Hybrid search (Phase 6.1-6.4)
- `feature/reranking-colbert` - ColBERT reranking (Phase 7.1-7.4)
- `feature/evaluation-ragas` - RAGAS evaluation (Phase 8.1-8.4)
- `feature/llm-integration` - LLM answer generation (Phase 9.1-9.4)
- `feature/optimization` - Quantization and caching (Phase 10.1-10.4)
- `bugfix/*` - Bug fixes (e.g., `bugfix/equation-boundary-detection`)
- `hotfix/*` - Critical production fixes

**Branch Naming Convention:**
```
<type>/<issue-number>-<short-description>

Examples:
feature/12-docling-parser
bugfix/45-chunking-split-equations
hotfix/78-qdrant-connection-timeout
```

### Commit Guidelines

**Commit Message Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build process, dependencies, tooling

**Examples:**
```bash
feat(parser): add Docling integration with LaTeX preservation

- Initialize Docling with optimal settings for equations
- Extract figures and maintain document hierarchy
- Implement graceful error handling
- Benchmark: 3.8s/page on test corpus

Closes #12

---

fix(chunking): prevent equation splitting across chunk boundaries

- Add regex detection for LaTeX equation environments
- Ensure equation + context stay together
- Add validation test for equation integrity

Fixes #45

---

test(retrieval): add golden set evaluation for hybrid search

- Create 200 manually curated test cases
- Validate Precision@5 > 80% threshold
- Document failure cases in issues

Related to #67

---

perf(embeddings): implement batch processing with GPU optimization

- Dynamic batch sizing based on text length
- Add progress tracking with tqdm
- Reduce embedding time by 60% (120s → 48s per 1000 chunks)

Closes #89
```

### Commit Cadence by Phase

**Phase 1 (Foundation):**
1. Initial commit after task 1.1: `chore: initialize project structure and dependencies`
2. After task 1.3: `feat(storage): deploy Qdrant with docker-compose`
3. After task 1.4: `docs: document corpus structure and file organization`

**Phase 2 (Document Processing):**
1. After task 2.1: `feat(parser): implement Docling parser with equation preservation`
2. After task 2.2: `feat(parser): add Marker fallback for high-speed parsing`
3. After task 2.3: `test(parser): validate 95%+ equation preservation accuracy`
4. **If validation finds major bugs**: Create issues before moving to Phase 3

**Phase 3-11:** Commit after each completed task (4-5 commits per phase)

**Validation Checkpoints:**
- After Phase 2: Commit + push, validate on CI
- After Phase 4: Commit + push, create release branch
- After Phase 7: Commit + push, tag as `v0.1.0-alpha`
- After Phase 10: Commit + push, tag as `v1.0.0-beta`

### Pull Request Workflow

**For Each Feature Branch:**
1. Create feature branch from `develop`
2. Implement feature with regular commits
3. Run tests locally: `pytest tests/`
4. Push branch: `git push -u origin feature/parser-docling`
5. Create PR: `feature/parser-docling` → `develop`
6. PR Description Template:
   ```markdown
   ## Description
   Implements Docling parser integration with LaTeX equation preservation.

   ## Changes
   - Added `src/parsers/docling_parser.py`
   - Created configuration file `config/docling.yaml`
   - Added unit tests in `tests/test_parsers.py`

   ## Testing
   - [x] Unit tests pass (15/15)
   - [x] Equation preservation: 96.2% on test set
   - [x] Benchmark: 3.7s/page average

   ## Issues
   - Closes #12
   - Relates to #8 (corpus organization)

   ## Validation Findings
   - Minor: Custom LaTeX macros not recognized (filed as #47)
   - Major: None
   ```
7. Review and merge to `develop`
8. Delete feature branch after merge

**PR Merge Strategy:**
- Use **Squash and Merge** for feature branches (clean history)
- Use **Merge Commit** for `develop` → `main` (preserve phase boundaries)

### Issue Tracking

**Issue Categories:**

**1. Bugs (Label: `bug`)**
```markdown
Title: [BUG] Chunking splits equations across boundaries

**Description:**
Semantic chunking occasionally splits LaTeX equation environments when
they appear near chunk size boundaries.

**Reproduction:**
1. Parse `Ch10_4P.tex`
2. Run semantic chunker with 750 token chunks
3. Observe equation on page 347 split between chunk_05 and chunk_06

**Expected Behavior:**
Equation should remain intact within single chunk with surrounding context.

**Actual Behavior:**
`\begin{equation}` in chunk_05, equation body in chunk_06

**Severity:** High (breaks equation rendering)
**Priority:** P0 (blocking Phase 3)

**Environment:**
- Python 3.11.5
- semchunk 2.0.1
- Test corpus: Aerospace Structures Chapter 10

**Validation Source:** Phase 2.3 validation testing
```

**2. Enhancement (Label: `enhancement`)**
```markdown
Title: [ENHANCEMENT] Add progress bar for batch embedding

**Description:**
Batch embedding processing for large corpus lacks progress feedback.

**Proposal:**
Add tqdm progress bar showing:
- Current batch / total batches
- Estimated time remaining
- Embeddings per second

**Benefit:**
Better user experience during long-running ingestion.

**Priority:** P2 (nice to have)
```

**3. Validation Finding (Label: `validation-finding`)**
```markdown
Title: [VALIDATION] Custom LaTeX macros not recognized by Docling

**Description:**
During Phase 2.3 validation, discovered that custom macros defined
in `AeroStructure-ERJohnson.cls` are not parsed correctly.

**Examples:**
- `\vect{F}` → renders as plain text instead of vector notation
- `\stress{\sigma}` → not converted to LaTeX

**Impact:** Minor (affects 12 of 500 test equations = 2.4%)
**Workaround:** Manual post-processing for affected equations

**Decision Needed:**
1. Extend Docling parser to handle custom macros
2. Preprocess LaTeX to expand macros before parsing
3. Accept as known limitation, document in README

**Phase:** 2.3 (Parser Validation)
**Blocking:** No (below 95% threshold)
```

**4. Task (Label: `task`)**
```markdown
Title: [TASK] Implement Matryoshka dimension reduction

**Description:**
Truncate Qwen3 embeddings from 768D to 256D to achieve 3x compression.

**Acceptance Criteria:**
- [ ] Generate full 768D embeddings
- [ ] Truncate to first 256 dimensions
- [ ] Validate 99.5%+ performance retention
- [ ] Update Qdrant schema to 256D

**Phase:** 4.2
**Estimated Time:** 2-3 hours
**Dependencies:** #34 (Qwen3 model setup)
```

**Issue Labels:**
- `bug` - Something broken
- `enhancement` - New feature or improvement
- `validation-finding` - Discovered during testing
- `task` - Implementation task
- `documentation` - Docs updates
- `performance` - Speed/memory optimization
- `P0` - Critical, blocking
- `P1` - High priority
- `P2` - Medium priority
- `P3` - Low priority
- `phase-1` through `phase-11` - Which phase
- `good-first-issue` - Easy for newcomers

### Git Commands by Phase

**Phase 1: Foundation**
```bash
# Initial setup
git checkout -b develop
git add pyproject.toml .gitignore README.md
git commit -m "chore: initialize project structure and dependencies"
git push -u origin develop

# Task 1.3: Qdrant setup
git add docker-compose.yml config/qdrant.yaml
git commit -m "feat(storage): deploy Qdrant with docker-compose

- Configure collection schema with sparse vectors
- Set HNSW parameters for optimal performance
- Add health check endpoint"
git push

# Task 1.4: Corpus documentation
git add data/corpus_manifest.json docs/corpus_structure.md
git commit -m "docs: document corpus structure and file organization"
git push
```

**Phase 2: Document Processing**
```bash
# Create feature branch
git checkout -b feature/12-parser-docling develop

# Task 2.1: Docling implementation
git add src/parsers/docling_parser.py config/docling.yaml
git commit -m "feat(parser): implement Docling parser with equation preservation

- Initialize Docling with optimal settings
- Extract figures and maintain hierarchy
- Implement graceful error handling
- Benchmark: 3.7s/page on test corpus

Related to #12"
git push -u origin feature/12-parser-docling

# Task 2.2: Marker fallback
git add src/parsers/marker_parser.py src/parsers/__init__.py
git commit -m "feat(parser): add Marker fallback for high-speed parsing

- Implement Marker with texify for equations
- Create parser selector based on speed priority
- Benchmark: 25 pages/second batch mode

Related to #12"
git push

# Task 2.3: Validation
git add tests/test_parsers.py data/evaluation/parser_validation.json
git commit -m "test(parser): validate 95%+ equation preservation accuracy

Results:
- Equation preservation: 96.2% (481/500 correct)
- Figure extraction: 100% (all 47 figures found)
- Hierarchy maintained: 100%

Validation findings:
- Minor issue with custom macros (filed as #47)

Closes #12"
git push

# Create PR, get review, merge to develop
# After merge:
git checkout develop
git pull
git branch -d feature/12-parser-docling
```

**After Each Phase:**
```bash
# Merge to develop and tag milestone
git checkout develop
git pull
git tag -a v0.2.0 -m "Phase 2 complete: Document processing pipeline"
git push origin v0.2.0
```

**MVP Release (After Phase 10):**
```bash
# Create release branch
git checkout -b release/v1.0.0 develop

# Final testing and bug fixes
# ... commits ...

# Merge to main
git checkout main
git merge --no-ff release/v1.0.0
git tag -a v1.0.0 -m "Release v1.0.0: Production-ready MVP

Features:
- Document parsing with 96%+ equation accuracy
- Hybrid retrieval with ColBERT reranking
- Binary quantization (32x compression)
- RAGAS evaluation framework
- Sub-2s end-to-end latency

Metrics:
- Precision@5: 82.3%
- Context Recall: 87.1%
- Faithfulness: 91.4%
"
git push origin main --tags

# Merge back to develop
git checkout develop
git merge --no-ff main
git push origin develop

# Delete release branch
git branch -d release/v1.0.0
```

### Issue Creation Triggers

**Automatic Issue Creation:**

**During Validation (Phase 2.3, 3.4, 4.4, 7.4, 8.3):**
- **Major bugs** (affect >5% of test cases or block progress):
  - Create `bug` issue with `P0` label
  - Stop current phase, fix immediately
  - Example: "Chunking splits 8% of equations" → Issue #45, fix before Phase 4

- **Minor bugs** (affect <5% of test cases, workaround exists):
  - Create `bug` issue with `P1` or `P2` label
  - Document workaround
  - Schedule fix for later
  - Example: "Custom macros not parsed (2.4%)" → Issue #47, fix in Phase 11

- **Performance issues** (exceed latency targets by >20%):
  - Create `performance` issue
  - Example: "Reranking latency 280ms (target: 200ms)" → Issue #89

**During Integration (Merging branches):**
- **Merge conflicts** requiring architecture decisions → `discussion` issue
- **Breaking changes** needing migration → `breaking-change` issue

**During Continuous Testing:**
- **Regression** in metrics (>5% drop from baseline) → `regression` issue with `P0`

### Example Issue Workflow

**Scenario: Validation finds equation splitting bug**

1. **Discovery (Phase 3.4 - Chunk Metadata Enrichment):**
   ```bash
   # Running validation
   pytest tests/test_chunking.py
   # FAILED: 41 of 500 equations split across chunks (8.2%)
   ```

2. **Create Issue:**
   ```markdown
   Title: [BUG] Semantic chunking splits 8.2% of equations across boundaries
   Labels: bug, P0, phase-3, validation-finding
   Milestone: Phase 3
   Assignee: (yourself or team member)
   ```

3. **Stop and Fix:**
   ```bash
   # Create bugfix branch from current feature branch
   git checkout -b bugfix/45-equation-splitting feature/chunking-semantic

   # Implement fix in src/chunking/equation_aware.py
   git add src/chunking/equation_aware.py tests/test_chunking.py
   git commit -m "fix(chunking): prevent equation splitting across chunks

   - Add regex detection for LaTeX equation environments
   - Look-ahead to check if equation fits in current chunk
   - Force new chunk boundary if equation would be split

   Validation results:
   - Equations split: 0 of 500 (0%)
   - New chunk count: 1,247 (was 1,203, +3.7%)

   Fixes #45"

   git push -u origin bugfix/45-equation-splitting
   ```

4. **Merge Fix:**
   ```bash
   # Merge bugfix into feature branch
   git checkout feature/chunking-semantic
   git merge --no-ff bugfix/45-equation-splitting
   git push

   # Delete bugfix branch
   git branch -d bugfix/45-equation-splitting
   ```

5. **Close Issue:**
   - GitHub automatically closes #45 when commit with "Fixes #45" merges to develop
   - Add comment: "Fixed and validated. All 500 test equations remain intact."

### .gitignore Configuration

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
*.egg-info/
dist/
build/

# Models (large files)
models/qwen3-8b/
models/jina-colbert-v2/
*.bin
*.safetensors
*.onnx

# Data
data/processed/
data/raw/*.pdf
*.vec
*.index

# Qdrant
qdrant_storage/
.qdrant/

# Evaluation outputs
outputs/
results/
benchmarks/*.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Secrets
.env
.env.local
*.key
config/*secret*

# Logs
*.log
logs/

# Cache
.cache/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Keep empty directories
!data/.gitkeep
!models/.gitkeep
!outputs/.gitkeep
```

### Git Hooks (Optional but Recommended)

**Pre-commit hook** (`.git/hooks/pre-commit`):
```bash
#!/bin/bash
# Run linting and formatting before commit

echo "Running pre-commit checks..."

# Format with Black
black src/ tests/
if [ $? -ne 0 ]; then
    echo "Black formatting failed"
    exit 1
fi

# Lint with Ruff
ruff check src/ tests/
if [ $? -ne 0 ]; then
    echo "Ruff linting failed"
    exit 1
fi

# Type check with mypy
mypy src/
if [ $? -ne 0 ]; then
    echo "Type checking failed"
    exit 1
fi

# Run fast unit tests
pytest tests/unit/ -v --tb=short
if [ $? -ne 0 ]; then
    echo "Unit tests failed"
    exit 1
fi

echo "Pre-commit checks passed!"
exit 0
```

**Pre-push hook** (`.git/hooks/pre-push`):
```bash
#!/bin/bash
# Run full test suite before pushing

echo "Running pre-push checks..."

# Run all tests including integration
pytest tests/ -v
if [ $? -ne 0 ]; then
    echo "Tests failed, push aborted"
    exit 1
fi

echo "All tests passed, proceeding with push"
exit 0
```

### GitHub Actions CI/CD (`.github/workflows/ci.yml`)

```yaml
name: CI Pipeline

on:
  push:
    branches: [develop, main]
  pull_request:
    branches: [develop, main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Lint with Ruff
      run: ruff check src/ tests/

    - name: Type check with mypy
      run: mypy src/

    - name: Run unit tests
      run: pytest tests/unit/ -v --cov=src --cov-report=xml

    - name: Run integration tests
      run: pytest tests/integration/ -v

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  validate:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'

    steps:
    - uses: actions/checkout@v3

    - name: Check commit messages
      run: |
        # Ensure commits follow conventional format
        python scripts/validate_commits.py

    - name: Check for large files
      run: |
        # Prevent committing large model files
        find . -type f -size +50M
```

## Implementation Tasks

### Phase 1: Project Foundation (Week 1)

#### 1.1 Project Setup
**Description**: Initialize Python project with modern tooling and structure
- Files to create:
  - `pyproject.toml` - Project metadata, dependencies, build configuration
  - `.gitignore` - Exclude models, data, virtual environments
  - `README.md` - Project overview and quick start
  - `.env.example` - Environment variable template
- Directory structure:
  ```
  aerospace-rag/
  ├── src/
  │   ├── parsers/         # Docling and Marker wrappers
  │   ├── chunking/        # Semantic chunking logic
  │   ├── embeddings/      # Qwen3 embedding generation
  │   ├── retrieval/       # Hybrid search and reranking
  │   ├── storage/         # Qdrant client and schema
  │   ├── evaluation/      # RAGAS integration
  │   └── utils/           # Logging, config, helpers
  ├── tests/               # Unit and integration tests
  ├── data/
  │   ├── raw/             # Original LaTeX files
  │   ├── processed/       # Parsed markdown
  │   └── evaluation/      # Golden test sets
  ├── models/              # Downloaded model weights
  ├── config/              # YAML configuration files
  └── notebooks/           # Jupyter notebooks for exploration
  ```
- Dependencies: `uv` or `poetry` for package management
- Estimated effort: 2-4 hours

#### 1.2 Environment Configuration
**Description**: Set up development environment with GPU support
- Install system dependencies:
  - Python 3.11+
  - Docker for Qdrant
  - CUDA 11.8+ (if GPU available)
- Create virtual environment
- Install core packages:
  ```toml
  docling >= 2.0.0
  qdrant-client >= 1.11.0
  sentence-transformers >= 3.0.0
  transformers >= 4.40.0
  torch >= 2.3.0
  semchunk >= 2.0.0
  ragas >= 0.1.0
  ```
- Configure environment variables for API keys, paths
- Dependencies: 1.1 complete
- Estimated effort: 1-2 hours

#### 1.3 Deploy Qdrant Vector Database
**Description**: Set up local Qdrant instance with optimal configuration
- Create `docker-compose.yml` for Qdrant service
- Configure collection schema:
  ```python
  {
    "vectors": {
      "size": 256,  # Matryoshka compressed
      "distance": "Cosine"
    },
    "sparse_vectors": {
      "text": {}  # BM25 for hybrid search
    },
    "payload_schema": {
      "document_id": "keyword",
      "section_path": "text",
      "chunk_type": "keyword",  # text, equation, figure
      "has_equations": "bool",
      "latex_source": "text",
      "page_number": "integer"
    }
  }
  ```
- Test connection and basic operations
- Implement health check endpoint
- Dependencies: 1.2 complete
- Estimated effort: 2-3 hours

#### 1.4 Document Corpus Preparation
**Description**: Organize Aerospace Structures LaTeX files for processing
- Survey Documents/Aerospace_Structures_LaTeX directory
- Identify chapter structure and dependencies
- Create document manifest with metadata
- Select 5 representative chapters for initial testing
- Document file structure and organization
- Dependencies: None
- Estimated effort: 2-3 hours

### Phase 2: Document Processing Pipeline (Week 1-2)

#### 2.1 Docling Parser Integration
**Description**: Implement wrapper for Docling with equation preservation
- Files to create: `src/parsers/docling_parser.py`
- Key functionality:
  - Initialize Docling with optimal settings for LaTeX
  - Parse LaTeX files to markdown with preserved equations
  - Extract figure bounding boxes and captions
  - Maintain document hierarchy (chapters, sections)
  - Handle parsing errors gracefully
- Configuration options:
  ```python
  DoclingConfig(
    force_ocr=False,
    extract_tables=True,
    extract_figures=True,
    output_format="markdown",
    preserve_latex=True
  )
  ```
- Test on 5 sample chapters
- Benchmark: 3.7 seconds/page target
- Dependencies: 1.2, 1.4 complete
- Estimated effort: 6-8 hours

#### 2.2 Marker Parser Fallback
**Description**: Implement Marker as high-speed alternative parser
- Files to create: `src/parsers/marker_parser.py`
- Key functionality:
  - Initialize Marker with texify for equations
  - Support batch processing mode (25 pages/second)
  - Enable LLM mode for enhanced table/math handling
  - Export to markdown and JSON formats
- Implement parser selector:
  ```python
  def select_parser(document, speed_priority=False):
      if speed_priority and len(document) > 50:
          return MarkerParser()
      return DoclingParser()
  ```
- Benchmark against Docling on accuracy and speed
- Dependencies: 2.1 complete
- Estimated effort: 4-6 hours

#### 2.3 Parser Output Validation
**Description**: Validate equation and figure preservation accuracy
- Files to create: `src/parsers/validator.py`
- Validation checks:
  - Equation count matches (LaTeX source vs. parsed output)
  - LaTeX syntax validity for extracted equations
  - Figure-caption associations preserved
  - No missing sections or broken hierarchy
- Create validation report template
- Test on 10 diverse LaTeX documents
- Target: 95%+ equation preservation
- Dependencies: 2.1, 2.2 complete
- Estimated effort: 4-5 hours

### Phase 3: Intelligent Chunking (Week 2)

#### 3.1 Semantic Chunking Implementation
**Description**: Implement semchunk with embedding similarity for boundary detection
- Files to create: `src/chunking/semantic_chunker.py`
- Key functionality:
  - Initialize semchunk with sentence-transformers tokenizer
  - Configure chunk size: 500-1000 tokens with 100 token overlap
  - Set similarity threshold at 80th percentile
  - Preserve sentence boundaries
- Integration with parser output
- Test different chunk sizes and overlaps
- Dependencies: 2.3 complete
- Estimated effort: 4-6 hours

#### 3.2 Hierarchy-Aware Chunking
**Description**: Extend semantic chunking to respect document structure
- Files to create: `src/chunking/hierarchical_chunker.py`
- Enhanced functionality:
  - Extract section hierarchy from Docling output
  - Never split across major section boundaries
  - Preserve subsection context in metadata
  - Track parent-child relationships between chunks
- Metadata enrichment:
  ```python
  {
    "section_path": "Chapter 10 > 10.3 > 10.3.2",
    "section_title": "Buckling Analysis",
    "parent_sections": ["Chapter 10", "10.3"],
    "depth_level": 3
  }
  ```
- Improvement target: 69.2% → 84.0% equivalence score
- Dependencies: 3.1 complete
- Estimated effort: 5-7 hours

#### 3.3 Equation Boundary Detection
**Description**: Prevent chunking from splitting mathematical content
- Files to create: `src/chunking/equation_aware.py`
- Key functionality:
  - Detect LaTeX equation blocks ($$...$$, \begin{equation})
  - Identify inline math ($...$)
  - Ensure equations remain intact within single chunks
  - Keep equation + context together (preceding/following sentences)
- Handle multi-line equations and equation arrays
- Validation: No split equations in output
- Dependencies: 3.2 complete
- Estimated effort: 3-4 hours

#### 3.4 Chunk Metadata Enrichment
**Description**: Add comprehensive metadata to each chunk for filtering and tracking
- Files to create: `src/chunking/metadata_enricher.py`
- Metadata fields:
  ```python
  {
    "document_id": "aerospace_structures_ch10",
    "section_path": "Chapter 10 > 10.3 > 10.3.2",
    "chunk_id": "ch10_s3_p2_chunk_05",
    "chunk_type": "text|equation|figure|mixed",
    "has_equations": True,
    "equation_count": 3,
    "figure_references": ["fig-10-15", "fig-10-16"],
    "keywords": ["buckling", "critical load", "Euler formula"],
    "page_number": 347,
    "tokens": 756,
    "latex_source": "original LaTeX if contains equations"
  }
  ```
- Extract technical keywords using spaCy or KeyBERT
- Dependencies: 3.3 complete
- Estimated effort: 3-4 hours

### Phase 4: Embedding Generation (Week 2-3)

#### 4.1 Qwen3-8B Model Setup
**Description**: Download and configure Qwen3 embedding model
- Files to create: `src/embeddings/qwen3_embedder.py`
- Setup tasks:
  - Download Qwen3-Embedding-8B from Hugging Face
  - Configure for GPU inference (if available)
  - Set up model caching directory
  - Implement batch processing for efficiency
- Configuration:
  ```python
  QwenConfig(
    model_name="Alibaba-NLP/gte-Qwen2-7B-instruct",
    max_length=32768,
    batch_size=32,
    normalize_embeddings=True,
    device="cuda" if available else "cpu"
  )
  ```
- Test embedding generation on sample text
- Dependencies: 1.2 complete
- Estimated effort: 3-4 hours

#### 4.2 Matryoshka Dimension Reduction
**Description**: Implement dimension reduction from 768 to 256 dimensions
- Extend `src/embeddings/qwen3_embedder.py`
- Functionality:
  - Generate full 768-dimension embeddings
  - Truncate to first 256 dimensions (Matryoshka property)
  - Validate performance retention (target: 99.5%)
- Create A/B test comparing 768D vs 256D on retrieval accuracy
- Storage savings: 3x compression
- Dependencies: 4.1 complete
- Estimated effort: 2-3 hours

#### 4.3 Batch Embedding Pipeline
**Description**: Implement efficient batch processing for large corpus
- Files to create: `src/embeddings/batch_processor.py`
- Key functionality:
  - Read chunks from processed documents
  - Batch processing with progress tracking
  - Error handling and retry logic
  - Checkpoint system for resumable processing
  - Cache embeddings to avoid recomputation
- Performance optimization:
  - Dynamic batch sizing based on text length
  - GPU utilization monitoring
  - Memory management for large batches
- Process full Aerospace Structures corpus
- Dependencies: 4.2, 3.4 complete
- Estimated effort: 5-6 hours

#### 4.4 Embedding Quality Validation
**Description**: Validate embedding quality through similarity tests
- Files to create: `tests/test_embeddings.py`
- Validation tests:
  - Similar technical concepts cluster together
  - Dissimilar concepts have low similarity
  - Mathematical equations with similar meaning group
  - Cross-language consistency (if applicable)
- Create visualization of embedding space (t-SNE/UMAP)
- Baseline comparison with OpenAI ada-002
- Dependencies: 4.3 complete
- Estimated effort: 3-4 hours

### Phase 5: Storage and Indexing (Week 3)

#### 5.1 Qdrant Collection Setup
**Description**: Create and configure Qdrant collection with optimal settings
- Files to create: `src/storage/qdrant_client.py`
- Implementation:
  - Create collection with vector and payload schema
  - Configure HNSW index parameters:
    ```python
    {
      "m": 16,  # number of edges per node
      "ef_construct": 200,  # quality of graph construction
      "full_scan_threshold": 10000
    }
    ```
  - Enable sparse vectors for BM25
  - Set up quantization (binary + int8)
- Test basic CRUD operations
- Dependencies: 1.3, 4.2 complete
- Estimated effort: 3-4 hours

#### 5.2 Batch Ingestion Pipeline
**Description**: Implement efficient pipeline to ingest embeddings into Qdrant
- Files to create: `src/storage/ingestion.py`
- Key functionality:
  - Batch upload points to Qdrant (1000 points/batch optimal)
  - Upsert strategy for updates
  - Progress tracking and logging
  - Error handling and rollback
  - Verify point count matches chunk count
- Ingestion workflow:
  ```python
  for batch in chunks:
      embeddings = embedder.embed(batch)
      points = create_points(batch, embeddings, metadata)
      qdrant.upsert(collection_name, points)
  ```
- Ingest full Aerospace Structures corpus
- Dependencies: 5.1, 4.3 complete
- Estimated effort: 4-5 hours

#### 5.3 Dual Index Architecture
**Description**: Implement separate indexes for semantic and exact equation matching
- Extend `src/storage/qdrant_client.py`
- Two-collection approach:
  1. **Semantic Collection**: All text with dense+sparse vectors
  2. **Equation Collection**: LaTeX strings with exact matching
- Equation index schema:
  ```python
  {
    "equation_id": "unique_identifier",
    "latex_source": "raw LaTeX string",
    "normalized_form": "simplified LaTeX for matching",
    "context_chunk_id": "link to semantic collection",
    "embedding": "semantic vector for similarity"
  }
  ```
- Implement cross-index linking
- Dependencies: 5.2 complete
- Estimated effort: 4-5 hours

#### 5.4 Metadata Filtering Setup
**Description**: Configure payload indexes for efficient filtering
- Extend `src/storage/qdrant_client.py`
- Create indexes on:
  - `document_id` (keyword)
  - `section_path` (text with tokenizer)
  - `chunk_type` (keyword)
  - `has_equations` (boolean)
  - `page_number` (integer range)
- Test filter performance:
  ```python
  filter = {
    "must": [
      {"key": "document_id", "match": {"value": "ch10"}},
      {"key": "has_equations", "match": {"value": True}}
    ]
  }
  ```
- Benchmark filtered vs unfiltered search latency
- Dependencies: 5.3 complete
- Estimated effort: 2-3 hours

### Phase 6: Hybrid Retrieval System (Week 3-4)

#### 6.1 BM25 Keyword Search Implementation
**Description**: Implement BM25 sparse vector search for technical terminology
- Files to create: `src/retrieval/bm25_retriever.py`
- Key functionality:
  - Configure Qdrant sparse vectors for BM25
  - Implement query preprocessing (lowercase, stemming optional)
  - Handle technical abbreviations and acronyms
  - Return top-K results with scores
- BM25 parameters:
  ```python
  {
    "k1": 1.2,  # term frequency saturation
    "b": 0.75    # length normalization
  }
  ```
- Test on technical term queries ("HNSW", "Euler buckling")
- Dependencies: 5.4 complete
- Estimated effort: 3-4 hours

#### 6.2 Semantic Vector Search
**Description**: Implement dense vector search using Qwen3 embeddings
- Files to create: `src/retrieval/semantic_retriever.py`
- Key functionality:
  - Embed query using same Qwen3 model
  - Search Qdrant with cosine similarity
  - Apply metadata filters from query analysis
  - Return top-K results with scores
- Query optimization:
  - Query expansion for ambiguous terms
  - Automatic filter detection (e.g., "in chapter 10")
- Test on conceptual queries
- Dependencies: 6.1 complete
- Estimated effort: 3-4 hours

#### 6.3 Reciprocal Rank Fusion (RRF)
**Description**: Combine BM25 and semantic results using RRF algorithm
- Files to create: `src/retrieval/fusion.py`
- Implementation:
  ```python
  def rrf_score(rank, k=60):
      return 1 / (k + rank)

  def fuse_results(bm25_results, semantic_results, alpha=0.7):
      # Combine scores with keyword weight alpha
      combined_scores = {}
      for doc_id, rank in bm25_results:
          combined_scores[doc_id] = alpha * rrf_score(rank)
      for doc_id, rank in semantic_results:
          combined_scores[doc_id] += (1-alpha) * rrf_score(rank)
      return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
  ```
- Tune α parameter:
  - α=0.7 for technical terminology heavy queries
  - α=0.5 for balanced queries
  - α=0.3 for conceptual queries
- A/B test against pure semantic search
- Target: 15-30% improvement in retrieval quality
- Dependencies: 6.2 complete
- Estimated effort: 3-4 hours

#### 6.4 Query Analysis and Routing
**Description**: Analyze queries to optimize retrieval strategy
- Files to create: `src/retrieval/query_analyzer.py`
- Query classification:
  - **Factual**: Direct answers from text (use keyword-heavy)
  - **Conceptual**: Understanding-based (use semantic-heavy)
  - **Equation**: Mathematical formula search (use equation index)
  - **Procedural**: Step-by-step (retrieve sequential chunks)
- Extract metadata filters from natural language:
  - "in chapter 10" → filter: document_id="ch10"
  - "with equations" → filter: has_equations=True
  - "about buckling" → keyword boost
- Dynamic α adjustment based on query type
- Dependencies: 6.3 complete
- Estimated effort: 4-5 hours

### Phase 7: Reranking Layer (Week 4)

#### 7.1 Jina-ColBERT-v2 Setup
**Description**: Download and configure ColBERT reranker
- Files to create: `src/retrieval/reranker.py`
- Setup tasks:
  - Download jinaai/jina-colbert-v2 model
  - Configure for efficient inference
  - Implement token packing for batch processing
- Model configuration:
  ```python
  ColBERTConfig(
    model_name="jinaai/jina-colbert-v2",
    max_length=8192,
    batch_size=16,
    device="cuda" if available else "cpu"
  )
  ```
- Test reranking on sample results
- Dependencies: 4.1 complete (similar setup)
- Estimated effort: 2-3 hours

#### 7.2 Two-Stage Retrieval Pipeline
**Description**: Implement retrieve-then-rerank architecture
- Extend `src/retrieval/reranker.py`
- Pipeline flow:
  1. Initial retrieval: Hybrid search returns top-100 candidates
  2. Reranking: ColBERT scores each candidate against query
  3. Selection: Return top-10 most relevant
- Performance optimization:
  - Cache reranking results for repeated queries
  - Batch reranking requests
  - Monitor latency (target: <200ms added latency)
- Implementation:
  ```python
  def retrieve_and_rerank(query, k_initial=100, k_final=10):
      # Stage 1: Fast retrieval
      candidates = hybrid_search(query, k=k_initial)

      # Stage 2: Precise reranking
      reranked = colbert_rerank(query, candidates)

      return reranked[:k_final]
  ```
- Dependencies: 7.1, 6.4 complete
- Estimated effort: 3-4 hours

#### 7.3 Reranking Performance Optimization
**Description**: Optimize reranker for production latency requirements
- Files to create: `src/retrieval/reranker_cache.py`
- Optimizations:
  - Result caching with Redis or in-memory LRU
  - Model quantization (FP16 or INT8)
  - Batch processing with dynamic batching
  - GPU memory management
- Benchmark configurations:
  - Latency at different k_initial values (50, 100, 200)
  - Impact of caching on repeated queries
  - GPU vs CPU inference speed
- Target: <200ms reranking latency
- Dependencies: 7.2 complete
- Estimated effort: 3-4 hours

#### 7.4 Reranking Quality Validation
**Description**: A/B test reranking impact on retrieval quality
- Files to create: `tests/test_reranking.py`
- Evaluation:
  - Precision@K with and without reranking
  - NDCG (Normalized Discounted Cumulative Gain)
  - Mean Reciprocal Rank (MRR)
  - User relevance judgments on 50 queries
- Create comparison report
- Target: 67% reduction in retrieval failure rate
- Dependencies: 7.3 complete
- Estimated effort: 4-5 hours

### Phase 8: Evaluation Framework (Week 4-5)

#### 8.1 RAGAS Integration
**Description**: Set up RAGAS for comprehensive RAG evaluation
- Files to create: `src/evaluation/ragas_evaluator.py`
- Key metrics:
  - **Context Precision**: Proportion of relevant chunks retrieved
  - **Context Recall**: Coverage of relevant information
  - **Faithfulness**: Answer grounded in retrieved context
  - **Answer Relevancy**: Answer addresses the query
- Configuration:
  ```python
  from ragas import evaluate
  from ragas.metrics import (
      context_precision,
      context_recall,
      faithfulness,
      answer_relevancy
  )

  results = evaluate(
      dataset=test_dataset,
      metrics=[context_precision, context_recall,
               faithfulness, answer_relevancy],
      llm=ChatOpenAI(model="gpt-4"),
      embeddings=QwenEmbeddings()
  )
  ```
- Dependencies: 7.4 complete (full pipeline)
- Estimated effort: 4-5 hours

#### 8.2 Synthetic Test Data Generation
**Description**: Use RAGAS to generate diverse question-answer pairs
- Extend `src/evaluation/ragas_evaluator.py`
- Generation process:
  1. Select representative chunks from corpus
  2. Generate questions at multiple complexity levels:
     - Simple: Direct factual questions
     - Reasoning: Require multi-step inference
     - Multi-context: Span multiple chunks
     - Conditional: "What if" scenarios
  3. Generate ground truth answers
  4. Human review and refinement
- Target: 500 question-answer pairs
- Cost estimate: ~$2.80 using Claude Haiku
- Time savings: 90% vs manual creation
- Dependencies: 8.1 complete
- Estimated effort: 6-8 hours (including review)

#### 8.3 Golden Test Set Creation
**Description**: Manually curate high-quality test cases for critical validation
- Files to create: `data/evaluation/golden_set.json`
- Test case categories:
  - Equation-heavy queries (50 cases)
  - Conceptual understanding (50 cases)
  - Cross-reference resolution (30 cases)
  - Figure interpretation (30 cases)
  - Procedural questions (40 cases)
- Each test case includes:
  ```json
  {
    "query": "What is the Euler buckling formula for columns?",
    "expected_chunks": ["ch10_s3_p2_chunk_05"],
    "expected_equations": ["P_cr = \\frac{\\pi^2 EI}{L^2}"],
    "expected_answer_contains": ["critical load", "elastic buckling"],
    "difficulty": "medium",
    "category": "equation"
  }
  ```
- Dependencies: 8.2 complete
- Estimated effort: 8-10 hours

#### 8.4 DeepEval CI/CD Integration
**Description**: Set up automated testing pipeline with DeepEval
- Files to create: `tests/test_rag_pipeline.py`, `.github/workflows/eval.yml`
- Integration:
  ```python
  from deepeval import assert_test
  from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
  from deepeval.test_case import LLMTestCase

  def test_rag_quality():
      test_case = LLMTestCase(
          input="What is Euler buckling?",
          actual_output=rag_system.query("What is Euler buckling?"),
          expected_output="Euler buckling describes...",
          retrieval_context=rag_system.get_contexts()
      )

      assert_test(test_case, [
          AnswerRelevancyMetric(threshold=0.7),
          FaithfulnessMetric(threshold=0.8)
      ])
  ```
- CI/CD workflow:
  - Run on every commit
  - Compare metrics against baseline
  - Fail build if metrics degrade >5%
- Dependencies: 8.3 complete
- Estimated effort: 4-5 hours

### Phase 9: LLM Integration and Answer Generation (Week 5)

#### 9.1 LLM Client Setup
**Description**: Configure LLM (GPT-4, Claude) for answer synthesis
- Files to create: `src/llm/client.py`
- Multi-provider support:
  - OpenAI GPT-4
  - Anthropic Claude
  - Local models via Ollama
- Configuration:
  ```python
  LLMConfig(
    provider="anthropic",  # or "openai", "ollama"
    model="claude-3-5-sonnet-20241022",
    temperature=0.1,  # Low for technical accuracy
    max_tokens=2000,
    timeout=30
  )
  ```
- Implement retry logic and fallbacks
- Dependencies: 1.2 complete
- Estimated effort: 3-4 hours

#### 9.2 Prompt Engineering for Technical Content
**Description**: Design prompts optimized for mathematical and technical responses
- Files to create: `src/llm/prompts.py`
- Prompt template:
  ```python
  TECHNICAL_QA_PROMPT = """You are an expert in aerospace engineering and structural mechanics. Answer the question based ONLY on the provided context.

Context:
{retrieved_chunks}

Equations referenced:
{latex_equations}

Question: {query}

Instructions:
1. Provide accurate technical information
2. Cite specific equations using LaTeX notation
3. Reference figure numbers when relevant
4. If information is insufficient, state this clearly
5. Maintain technical precision in terminology

Answer:"""
  ```
- Specialized prompts for:
  - Equation explanation
  - Procedural steps
  - Conceptual understanding
  - Comparative analysis
- Dependencies: 9.1 complete
- Estimated effort: 4-5 hours

#### 9.3 Citation and Source Tracking
**Description**: Implement citation system linking answers to source chunks
- Files to create: `src/llm/citation_tracker.py`
- Functionality:
  - Track which chunks contributed to answer
  - Generate citations in standardized format
  - Link equations to source pages
  - Provide "View Source" links
- Citation format:
  ```python
  {
    "answer": "Euler buckling occurs when...",
    "citations": [
      {
        "chunk_id": "ch10_s3_p2_chunk_05",
        "section": "Chapter 10, Section 10.3.2",
        "page": 347,
        "relevance_score": 0.94
      }
    ],
    "equations_used": ["\\frac{\\pi^2 EI}{L^2}"],
    "confidence": 0.87
  }
  ```
- Dependencies: 9.2 complete
- Estimated effort: 4-5 hours

#### 9.4 Streaming Response Implementation
**Description**: Enable streaming for real-time answer generation
- Extend `src/llm/client.py`
- Implementation:
  - Stream tokens as generated
  - Update citations progressively
  - Handle interruptions gracefully
- Example:
  ```python
  async def stream_answer(query):
      contexts = await retrieve_and_rerank(query)
      async for chunk in llm.stream(prompt, contexts):
          yield {
            "type": "token",
            "content": chunk,
            "citations": extract_citations(chunk)
          }
  ```
- Dependencies: 9.3 complete
- Estimated effort: 3-4 hours

### Phase 10: Optimization and Production Readiness (Week 5-6)

#### 10.1 Binary Quantization Implementation
**Description**: Apply binary quantization for 32x storage compression
- Files to create: `src/storage/quantization.py`
- Implementation:
  - Convert float32 embeddings to binary (1-bit)
  - Keep int8 version for rescoring
  - Update Qdrant collection configuration:
    ```python
    {
      "quantization_config": {
        "scalar": {
          "type": "int8",
          "quantile": 0.99,
          "always_ram": True
        },
        "binary": {
          "always_ram": False
        }
      }
    }
    ```
- Test accuracy retention:
  - Binary direct: ~92.5% performance
  - Binary + int8 rescore: ~96% performance
- Storage reduction: 200GB → 6.25GB for embeddings
- Dependencies: 5.2 complete
- Estimated effort: 4-5 hours

#### 10.2 Result Caching Layer
**Description**: Implement caching for frequent queries and expensive operations
- Files to create: `src/utils/cache.py`
- Multi-level caching:
  1. **Query cache**: Hash query → cached results (1 hour TTL)
  2. **Embedding cache**: Text → embedding vector (persistent)
  3. **Reranking cache**: (query, doc_id) → score (24 hour TTL)
- Cache backend: Redis for distributed or LRU for local
- Cache warming: Pre-compute common technical term queries
- Monitoring: Track hit rates and latency improvements
- Dependencies: 7.3, 9.4 complete
- Estimated effort: 4-5 hours

#### 10.3 Performance Benchmarking
**Description**: Comprehensive performance testing across all components
- Files to create: `tests/benchmark_performance.py`
- Benchmarks:
  - Document parsing: pages/second
  - Embedding generation: tokens/second
  - Retrieval latency: p50, p95, p99
  - Reranking latency: per query
  - End-to-end: query to answer time
  - Memory usage: peak and average
  - Storage: GB per million chunks
- Targets:
  - Parsing: 3.7s/page (Docling) or 0.04s/page (Marker)
  - Retrieval: <100ms
  - Reranking: <200ms
  - End-to-end: <2s including LLM
- Create performance report dashboard
- Dependencies: All previous phases
- Estimated effort: 6-8 hours

#### 10.4 Monitoring and Logging
**Description**: Implement production monitoring with Langfuse
- Files to create: `src/utils/monitoring.py`
- Langfuse integration:
  ```python
  from langfuse import Langfuse

  langfuse = Langfuse(
    public_key="pk_...",
    secret_key="sk_..."
  )

  @langfuse.observe()
  def query_rag(query: str):
      with langfuse.span(name="retrieval") as span:
          contexts = retrieve_and_rerank(query)
          span.log({"num_contexts": len(contexts)})

      with langfuse.span(name="generation") as span:
          answer = llm.generate(query, contexts)
          span.log({"tokens": count_tokens(answer)})

      return answer
  ```
- Metrics tracked:
  - Query volume and patterns
  - Latency breakdown by component
  - Error rates and types
  - Token usage and costs
  - Quality metrics (RAGAS scores)
- Alert on degradations
- Dependencies: 10.3 complete
- Estimated effort: 4-5 hours

### Phase 11: Advanced Features (Week 6+, Optional)

#### 11.1 Fine-Tuned Domain Embeddings
**Description**: Fine-tune Qwen3 on Aerospace Structures corpus
- Files to create: `src/embeddings/fine_tuning.py`
- Process:
  1. Generate 5K-10K synthetic Q&A pairs using RAGAS
  2. Create training pairs: (question, relevant_chunk)
  3. Fine-tune using sentence-transformers:
     ```python
     from sentence_transformers import SentenceTransformer
     from sentence_transformers.losses import MultipleNegativesRankingLoss

     model = SentenceTransformer("Qwen3-8B")
     train_dataset = create_training_pairs(qa_pairs)
     model.fit(
         train_objectives=[(train_dataloader, MultipleNegativesRankingLoss(model))],
         epochs=3,
         warmup_steps=100
     )
     ```
  4. Evaluate improvement on test set
- Expected gain: 6-10% on domain-specific queries
- Cost: 3 minutes on consumer GPU
- Dependencies: 8.2, 10.3 complete
- Estimated effort: 8-10 hours

#### 11.2 Equation Dependency Graph
**Description**: Build knowledge graph of mathematical relationships
- Files to create: `src/knowledge_graph/equation_graph.py`
- Graph structure:
  - Nodes: Equations, variables, theorems
  - Edges: "derives from", "uses", "special case of"
- Extraction:
  - Parse LaTeX to identify variables and relationships
  - Use LLM to identify conceptual connections
  - Track citations and references
- Query enhancement:
  - Expand equation queries to related concepts
  - Traverse graph for multi-hop reasoning
- Dependencies: 5.3, 9.3 complete
- Estimated effort: 12-15 hours

#### 11.3 Multi-File Reference Resolution
**Description**: Handle cross-chapter citations and references
- Files to create: `src/parsers/reference_resolver.py`
- Functionality:
  - Extract citation patterns (\ref{}, \cite{})
  - Build cross-reference index
  - Resolve references during retrieval
  - Return cited content alongside primary results
- Example:
  ```python
  # Query mentions "as shown in Section 8.3"
  # System retrieves:
  # 1. Primary context (current section)
  # 2. Referenced content (Section 8.3)
  # 3. Linking context explaining relationship
  ```
- Dependencies: 2.3, 5.3 complete
- Estimated effort: 8-10 hours

#### 11.4 CLIP Visual Embeddings for Figures
**Description**: Enable figure similarity search using CLIP
- Files to create: `src/embeddings/clip_embedder.py`
- Implementation:
  - Extract figure images from LaTeX compilation
  - Generate CLIP embeddings for visual content
  - Store in separate Qdrant collection
  - Cross-link with text embeddings
- Use cases:
  - "Find similar diagrams"
  - "Show stress distribution figures"
  - Visual question answering
- Dependencies: 2.1, 4.3 complete
- Estimated effort: 10-12 hours

## Codebase Integration Points

### Files to Create
**Core Infrastructure:**
- `pyproject.toml` - Project metadata and dependencies
- `src/parsers/docling_parser.py` - Docling integration
- `src/parsers/marker_parser.py` - Marker integration
- `src/parsers/validator.py` - Parsing validation
- `src/chunking/semantic_chunker.py` - Semchunk integration
- `src/chunking/hierarchical_chunker.py` - Hierarchy-aware chunking
- `src/chunking/equation_aware.py` - Equation boundary detection
- `src/chunking/metadata_enricher.py` - Metadata generation
- `src/embeddings/qwen3_embedder.py` - Qwen3 embedding model
- `src/embeddings/batch_processor.py` - Batch embedding generation
- `src/storage/qdrant_client.py` - Qdrant integration
- `src/storage/ingestion.py` - Batch ingestion pipeline
- `src/storage/quantization.py` - Binary/int8 quantization
- `src/retrieval/bm25_retriever.py` - BM25 keyword search
- `src/retrieval/semantic_retriever.py` - Dense vector search
- `src/retrieval/fusion.py` - RRF hybrid fusion
- `src/retrieval/query_analyzer.py` - Query classification
- `src/retrieval/reranker.py` - ColBERT reranking
- `src/llm/client.py` - Multi-provider LLM client
- `src/llm/prompts.py` - Prompt templates
- `src/llm/citation_tracker.py` - Citation management
- `src/evaluation/ragas_evaluator.py` - RAGAS integration
- `src/utils/cache.py` - Caching layer
- `src/utils/monitoring.py` - Langfuse monitoring
- `src/utils/config.py` - Configuration management
- `src/utils/logger.py` - Logging utilities

**Configuration:**
- `config/docling.yaml` - Docling parser settings
- `config/qdrant.yaml` - Vector DB configuration
- `config/models.yaml` - Model paths and settings
- `config/retrieval.yaml` - Retrieval parameters
- `docker-compose.yml` - Qdrant deployment

**Testing:**
- `tests/test_parsers.py` - Parser unit tests
- `tests/test_chunking.py` - Chunking unit tests
- `tests/test_embeddings.py` - Embedding quality tests
- `tests/test_retrieval.py` - Retrieval accuracy tests
- `tests/test_reranking.py` - Reranking quality tests
- `tests/test_rag_pipeline.py` - End-to-end tests
- `tests/benchmark_performance.py` - Performance benchmarks

**Data and Models:**
- `data/raw/` - Original LaTeX files
- `data/processed/` - Parsed markdown
- `data/evaluation/golden_set.json` - Manual test cases
- `data/evaluation/synthetic_set.json` - RAGAS generated tests
- `models/qwen3-8b/` - Downloaded model weights
- `models/jina-colbert-v2/` - Reranker model

### Existing Patterns to Follow
This is a greenfield project, but establish patterns early:
- **Error Handling**: Use custom exceptions with meaningful messages
- **Logging**: Structured logging with contextual information
- **Configuration**: YAML files for complex config, environment variables for secrets
- **Testing**: Pytest with fixtures for common setups
- **Documentation**: Docstrings in Google format, type hints throughout
- **Code Style**: Black formatter, ruff linter, mypy type checking

## Technical Design

### Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Query Analyzer                              │
│  • Classify query type (factual/conceptual/equation)           │
│  • Extract metadata filters                                    │
│  • Route to appropriate retrieval strategy                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Hybrid Retrieval (Top-100)                     │
│  ┌──────────────────┐              ┌──────────────────────┐    │
│  │  BM25 Search     │              │  Semantic Search     │    │
│  │  (Sparse Vector) │              │  (Dense Vector)      │    │
│  └────────┬─────────┘              └──────────┬───────────┘    │
│           │                                    │                │
│           └────────────┬───────────────────────┘                │
│                        ▼                                        │
│             Reciprocal Rank Fusion (RRF)                        │
│             α=0.7 for technical queries                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              ColBERT Reranker (Top-100 → Top-10)                │
│  • Late-interaction scoring                                    │
│  • Query-document attention                                    │
│  • Return highest relevance chunks                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LLM Answer Generation                         │
│  • Synthesize answer from contexts                             │
│  • Maintain LaTeX notation                                     │
│  • Track citations                                             │
│  • Stream response                                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Formatted Response                           │
│  • Answer text with LaTeX                                      │
│  • Source citations (section, page)                            │
│  • Confidence score                                            │
│  • Related equations                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

**Ingestion Flow:**
```
LaTeX Files
    ↓
Docling Parser (3.7s/page)
    ↓
Markdown + Equations + Figures
    ↓
Hierarchical Chunking (500-1000 tokens)
    ↓
Chunks + Rich Metadata
    ↓
Qwen3-8B Embeddings (256D, Matryoshka)
    ↓
Qdrant Storage (Binary + Int8 Quantized)
```

**Query Flow:**
```
Natural Language Query
    ↓
Query Analysis (type, filters)
    ↓
Embed Query (Qwen3-8B)
    ↓
┌─────────────┬──────────────┐
│             │              │
BM25 Search   Semantic      Equation
(keyword)     Search         Exact Match
│             │              │
└─────────────┴──────────────┘
    ↓
Fusion (RRF, α=0.7)
    ↓
Top-100 Candidates
    ↓
ColBERT Reranking
    ↓
Top-10 Best Matches
    ↓
LLM Synthesis
    ↓
Answer + Citations
```

### API Design (Future Web Interface)

**RESTful Endpoints:**
```
POST /api/v1/query
{
  "query": "What is Euler buckling?",
  "filters": {
    "document_id": "ch10",
    "has_equations": true
  },
  "top_k": 5,
  "include_citations": true
}

Response:
{
  "answer": "Euler buckling describes...",
  "confidence": 0.87,
  "citations": [
    {
      "chunk_id": "ch10_s3_p2_chunk_05",
      "section": "Chapter 10, Section 10.3.2",
      "page": 347,
      "relevance": 0.94,
      "excerpt": "The critical load for elastic buckling..."
    }
  ],
  "equations": [
    "P_{cr} = \\frac{\\pi^2 EI}{L^2}"
  ],
  "latency_ms": 856
}

GET /api/v1/documents
# List all indexed documents

GET /api/v1/documents/{doc_id}/sections
# Get document structure

POST /api/v1/embed
# Generate embedding for text

GET /api/v1/health
# System health check
```

## Dependencies and Libraries

**Core Dependencies:**
```toml
[project]
name = "aerospace-rag"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # Document Processing
    "docling>=2.0.0",
    "marker-pdf>=0.3.0",
    "pypdf>=4.0.0",

    # Vector Database
    "qdrant-client>=1.11.0",

    # Embeddings & Models
    "sentence-transformers>=3.0.0",
    "transformers>=4.40.0",
    "torch>=2.3.0",
    "accelerate>=0.29.0",

    # Chunking
    "semchunk>=2.0.0",
    "spacy>=3.7.0",

    # Retrieval
    "rank-bm25>=0.2.2",

    # LLM Integration
    "openai>=1.30.0",
    "anthropic>=0.25.0",

    # Evaluation
    "ragas>=0.1.0",
    "deepeval>=0.21.0",

    # Monitoring
    "langfuse>=2.0.0",

    # Utilities
    "pydantic>=2.7.0",
    "pyyaml>=6.0",
    "redis>=5.0.0",
    "tqdm>=4.66.0",
    "loguru>=0.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "black>=24.0.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]

viz = [
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "plotly>=5.20.0",
]

notebooks = [
    "jupyter>=1.0.0",
    "ipykernel>=6.29.0",
]
```

**System Requirements:**
- Python 3.11+
- Docker & Docker Compose (for Qdrant)
- CUDA 11.8+ (optional, for GPU acceleration)
- 32GB+ RAM (16GB minimum without GPU)
- 100GB+ disk space (for models and data)

## Testing Strategy

### Unit Tests
**Component-Level Testing:**
- `test_parsers.py`: Validate parsing accuracy on diverse LaTeX files
- `test_chunking.py`: Ensure chunks respect boundaries, test metadata
- `test_embeddings.py`: Verify embedding quality, test Matryoshka truncation
- `test_storage.py`: Test CRUD operations, quantization accuracy
- `test_retrieval.py`: Validate BM25, semantic, and fusion logic
- `test_reranking.py`: Verify ColBERT scoring correctness

### Integration Tests
**End-to-End Workflows:**
- `test_ingestion_pipeline.py`: Full document → Qdrant flow
- `test_query_pipeline.py`: Query → retrieval → rerank → LLM
- `test_error_handling.py`: Graceful degradation scenarios
- `test_concurrent_queries.py`: Load testing with parallel requests

### Evaluation Tests
**Quality Assurance:**
- `test_golden_set.py`: Run against 200 manually curated test cases
- `test_synthetic_set.py`: Evaluate on 500 RAGAS-generated questions
- `test_regression.py`: Compare metrics against baseline after changes
- Metrics tracked:
  - Retrieval: Precision@5, Recall@10, NDCG
  - Generation: Faithfulness, Answer Relevancy, Context Precision
  - Performance: Latency p50/p95/p99, Throughput (queries/second)

### Continuous Testing
- Run unit tests on every commit
- Run integration tests on PR creation
- Run full evaluation suite weekly
- Performance benchmarks on release candidates

## Success Criteria

### Must Have (MVP - Week 6)
- ✅ Parse LaTeX documents with 95%+ equation preservation accuracy
- ✅ Implement hybrid search (BM25 + semantic) with RRF fusion
- ✅ Deploy ColBERT reranking achieving <200ms latency
- ✅ End-to-end query latency <2 seconds (including LLM)
- ✅ Achieve Precision@5 > 80% on golden test set
- ✅ Index full Aerospace Structures corpus (18 chapters)
- ✅ Generate 500 synthetic test cases with RAGAS
- ✅ Implement binary quantization for 32x storage reduction

### Nice to Have (V1.1 - Week 8)
- ✅ Fine-tuned Qwen3 embeddings on domain corpus (6-10% improvement)
- ✅ Langfuse monitoring dashboard with real-time metrics
- ✅ Result caching reducing latency by 50% for repeated queries
- ✅ Cross-chapter reference resolution
- ✅ Streaming responses with progressive citations
- ✅ DeepEval CI/CD integration for regression testing

### Stretch Goals (V2.0 - Week 12+)
- ✅ Equation dependency knowledge graph
- ✅ CLIP-based figure similarity search
- ✅ Interactive web UI with Gradio
- ✅ Multi-language support (beyond English)
- ✅ Incremental indexing for document updates
- ✅ Collaborative features (shared annotations, highlights)

### Quantitative Targets

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Equation Preservation** | >95% | Manual review of 100 equations |
| **Retrieval Precision@5** | >80% | RAGAS evaluation on golden set |
| **Context Recall** | >85% | RAGAS metric on test set |
| **Answer Faithfulness** | >90% | RAGAS faithfulness score |
| **Query Latency (p95)** | <2s | Performance benchmarks |
| **Retrieval Latency** | <100ms | Qdrant search timing |
| **Reranking Latency** | <200ms | ColBERT timing |
| **Storage Efficiency** | 32x compression | Binary quantization vs float32 |
| **System Uptime** | >99% | Monitoring logs (if deployed) |

## Notes and Considerations

### Critical Success Factors
1. **Equation Preservation is Non-Negotiable**: LaTeX fidelity is the primary differentiator for this system
2. **Chunking Quality > Retrieval Algorithms**: Poor chunking cannot be fixed by better retrieval
3. **Benchmark Early and Often**: Establish baselines before optimization to measure true impact
4. **Domain-Specific Tuning Matters**: Generic embeddings won't capture technical nuances
5. **User Feedback Loop**: Even synthetic evaluation needs human validation

### Potential Challenges

**Technical Challenges:**
- **LaTeX Parsing Edge Cases**: Complex equation arrays, custom macros, nested environments
  - *Mitigation*: Test on diverse samples, maintain fallback to Marker, allow manual corrections
- **Chunking Boundary Detection**: Proofs and derivations span multiple paragraphs
  - *Mitigation*: Use LLM-based agentic chunking for critical sections, tune overlap
- **GPU Memory Constraints**: Large models (Qwen3-8B, ColBERT) may exceed VRAM
  - *Mitigation*: Implement model quantization (FP16, INT8), batch size tuning, CPU fallback
- **Cross-Reference Complexity**: Aerospace textbook has extensive chapter cross-references
  - *Mitigation*: Build reference index in Phase 11, initially return context without resolution

**Performance Challenges:**
- **Reranking Latency**: ColBERT can be slow for large candidate sets
  - *Mitigation*: Aggressive caching, reduce k_initial to 50, model quantization
- **Storage Costs**: Full-precision embeddings for entire corpus is expensive
  - *Mitigation*: Binary quantization from start, validate accuracy tradeoff
- **Cold Start**: First query slow due to model loading
  - *Mitigation*: Model pre-loading, warm-up queries on startup

**Quality Challenges:**
- **Hallucination Risk**: LLM may generate plausible but incorrect technical information
  - *Mitigation*: Strict faithfulness monitoring, low temperature (0.1), citation enforcement
- **Evaluation Validity**: Synthetic tests may not reflect real user needs
  - *Mitigation*: Combine synthetic (500) with golden (200) manual tests, user acceptance testing
- **Retrieval Bias**: System may favor frequently accessed content
  - *Mitigation*: Monitor diversity metrics, implement query understanding to route to right sections

### Future Enhancements

**Short-Term (3-6 months):**
- Incremental indexing for document updates
- Multi-modal search (combine text query + example figure)
- Query suggestion and auto-complete
- Export answers to LaTeX/PDF/Markdown

**Medium-Term (6-12 months):**
- Support additional textbooks and manuals
- Comparative analysis across multiple sources
- Interactive equation manipulation
- Mobile app for on-the-go access

**Long-Term (12+ months):**
- Automated homework problem generation
- Proof verification and checking
- Integration with CAD/FEA software
- Collaborative learning platform features

### Development Philosophy
- **Start Simple, Optimize Later**: Get basic pipeline working before adding complexity
- **Measure Everything**: If you can't measure it, you can't improve it
- **Fail Fast**: Test each component in isolation before integration
- **Document as You Go**: Don't leave documentation for the end
- **Iterate on Feedback**: Build feedback loops early (evaluation, monitoring)

---

## Execution Checklist

**Before Starting:**
- [ ] Read and understand this full plan
- [ ] Set up development environment (Python, Docker, GPU drivers)
- [ ] Clone/download Aerospace Structures LaTeX files
- [ ] Create project repository with .gitignore
- [ ] Review budget for LLM API costs (estimate $50-100 for evaluation)
- [ ] **Initialize Git repository and create `develop` branch**
- [ ] **Set up GitHub repository and configure branch protection rules**

**Weekly Checkpoints (with Git workflow):**
- [ ] **Week 1**:
  - Project setup, Qdrant deployed, corpus organized, Docling tested
  - Git: 3-5 commits, `develop` branch established, initial PR merged
  - Tag: `v0.1.0` (Foundation complete)

- [ ] **Week 2**:
  - Parsing pipeline complete, chunking implemented, initial ingestion done
  - Git: Feature branches for parser and chunking, 8-10 commits
  - Validation: File issues for any bugs found (major: P0, minor: P1/P2)
  - Tag: `v0.2.0` (Document processing complete)

- [ ] **Week 3**:
  - Embeddings generated, retrieval working, basic search functional
  - Git: Feature branches for embeddings and retrieval, 8-10 commits
  - Tag: `v0.3.0` (Retrieval system functional)

- [ ] **Week 4**:
  - Reranking integrated, LLM connected, evaluation framework setup
  - Git: Feature branches for reranking and evaluation, 8-10 commits
  - Validation: Run full evaluation suite, file performance issues if latency >20% over target
  - Tag: `v0.4.0-alpha` (End-to-end pipeline working)

- [ ] **Week 5**:
  - Optimization applied (quantization, caching), monitoring deployed
  - Git: Feature branch for optimization, 5-7 commits
  - Tag: `v0.9.0-beta` (Optimization complete)

- [ ] **Week 6**:
  - Full testing complete, documentation written, MVP ready
  - Git: Create release branch `release/v1.0.0`
  - Final testing and bug fixes on release branch
  - Merge to `main` and tag `v1.0.0`
  - Push to GitHub: `git push origin main --tags`

**Success Validation:**
- [ ] Run full test suite (unit + integration + evaluation)
- [ ] Verify all quantitative targets met
- [ ] Conduct user acceptance testing (5-10 test queries)
- [ ] Review performance benchmarks
- [ ] Check monitoring dashboard shows healthy metrics
- [ ] **Review open issues: All P0 resolved, P1 documented with workarounds**
- [ ] **Clean commit history: Check that all commits follow conventional format**
- [ ] **CI/CD passing: All GitHub Actions workflows green**

---

*This plan is ready for execution using `/execute-plan PRPs/aerospace-latex-rag.md`*

**Estimated Total Effort:** 200-250 hours (6-8 weeks full-time, 12-16 weeks part-time)

**Key Milestones:**
1. **Week 2**: First successful end-to-end query
2. **Week 4**: Passing evaluation thresholds
3. **Week 6**: Production-ready MVP

**Next Steps:**
1. Review this plan and adjust priorities based on your goals
2. Set up development environment (Phase 1.2)
3. Start with Quick Win: Test Docling on sample chapter (Phase 2.1)
4. Begin systematic execution following phase order
