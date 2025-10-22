# Development Guide

## Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/tech-doc-scanner.git
cd tech-doc-scanner

# Install with dev dependencies
uv sync

# Activate virtual environment (if needed)
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

## Project Structure

```
tech-doc-scanner/
├── src/tech_doc_scanner/     # Main package
│   ├── __init__.py           # Public API
│   ├── config.py             # Configuration system
│   ├── converter.py          # Document converter wrapper
│   ├── pipelines.py          # Pipeline factory
│   ├── cache.py              # Model caching
│   ├── stats.py              # Statistics collection
│   ├── cli.py                # CLI implementation
│   ├── latex/                # LaTeX processing
│   │   ├── cleaner.py        # LaTeX cleaning
│   │   └── validator.py      # LaTeX validation
│   └── enrichment/           # Enrichment models
│       ├── base.py           # Base classes
│       └── pix2text.py       # Pix2Text integration
├── tests/                    # Test suite
├── examples/                 # Usage examples
├── docs/                     # Documentation
└── pyproject.toml            # Package configuration
```

## Running Tests

### All Tests

```bash
pytest tests/ -v
```

### With Coverage

```bash
pytest tests/ --cov=src/tech_doc_scanner --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Specific Test Files

```bash
pytest tests/test_latex_cleaner.py -v
pytest tests/test_integration.py::TestEndToEndIntegration -v
```

### Slow Tests

Some tests are marked as slow (e.g., end-to-end conversions):

```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run only slow tests
pytest tests/ -m "slow"
```

## Code Quality

### Linting

```bash
# Ruff (fast Python linter)
ruff check src/ tests/

# Auto-fix
ruff check --fix src/ tests/
```

### Formatting

```bash
# Black
black src/ tests/

# Check without modifying
black --check src/ tests/
```

### Type Checking

```bash
# MyPy
mypy src/
```

## Adding New Features

### 1. Create Branch

```bash
git checkout -b feature/my-new-feature
```

### 2. Write Tests First (TDD)

```python
# tests/test_my_feature.py
def test_my_new_feature():
    """Test the new feature."""
    result = my_new_feature()
    assert result == expected
```

### 3. Implement Feature

```python
# src/tech_doc_scanner/my_module.py
def my_new_feature():
    """Implement the feature."""
    ...
```

### 4. Run Tests

```bash
pytest tests/test_my_feature.py -v
```

### 5. Update Documentation

- Add docstrings to code
- Update relevant docs in `docs/`
- Add example if applicable

### 6. Commit and Push

```bash
git add .
git commit -m "Add my new feature

- Implemented X
- Added tests
- Updated documentation"

git push origin feature/my-new-feature
```

## Adding New Enrichment Models

To add a new enrichment model (e.g., for diagrams, tables):

### 1. Create Enrichment Class

```python
# src/tech_doc_scanner/enrichment/my_enrichment.py
from docling.datamodel.base_models import BaseItemAndImageEnrichmentModel

class MyEnrichmentModel(BaseItemAndImageEnrichmentModel):
    def __init__(self, config):
        super().__init__(enabled=config.enabled)
        self.config = config
    
    def is_processable(self, item, doc_item):
        """Check if item should be processed."""
        return item.label == "TARGET_TYPE"
    
    def __call__(self, item, image):
        """Process the item."""
        # Your processing logic
        ...
```

### 2. Add to Pipeline

```python
# src/tech_doc_scanner/pipelines.py
class MyCustomPipeline(StandardPdfPipeline):
    def __init__(self, docling_config, my_config):
        super().__init__(pipeline_options=...)
        
        # Add your enrichment
        self.enrichment = MyEnrichmentModel(my_config)
```

### 3. Add Tests

```python
# tests/test_my_enrichment.py
def test_my_enrichment():
    enrichment = MyEnrichmentModel(config)
    result = enrichment(item, image)
    assert result is not None
```

## Debugging

### Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use CLI flag:

```bash
tech-doc-scanner convert input.pdf --verbose
```

### Inspect Intermediate Results

```python
# In converter.py or enrichment model
import json
from pathlib import Path

# Save intermediate results
debug_path = Path("debug_output.json")
debug_path.write_text(json.dumps(intermediate_data, indent=2))
```

### Python Debugger

```python
import pdb

# Set breakpoint
pdb.set_trace()

# Or use breakpoint() in Python 3.7+
breakpoint()
```

## Performance Profiling

### Time Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = converter.convert_file(pdf_path, output_dir)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def my_function():
    # Your code
    ...
```

## Contributing Guidelines

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions focused and small

### Commit Messages

Format:
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement

Example:
```
feat: Add GPU acceleration support

- Implement CUDA device selection
- Add MPS support for Apple Silicon
- Update configuration schema

Closes #123
```

### Pull Request Process

1. Create feature branch
2. Write tests
3. Implement feature
4. Update documentation
5. Run full test suite
6. Create PR with description
7. Address review comments
8. Merge when approved

## Release Process

### 1. Version Bump

Update version in `src/tech_doc_scanner/__init__.py`:

```python
__version__ = "0.2.0"
```

### 2. Update Changelog

Document changes in `CHANGELOG.md`.

### 3. Tag Release

```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

### 4. Build Package

```bash
python -m build
```

### 5. Publish (if applicable)

```bash
python -m twine upload dist/*
```

## Troubleshooting Development

### Import Errors

Ensure package is installed in editable mode:

```bash
pip install -e .
# or
uv sync
```

### Test Failures

```bash
# Run with verbose output
pytest tests/ -vv

# Show print statements
pytest tests/ -s

# Stop on first failure
pytest tests/ -x
```

### Type Errors

```bash
# Check specific file
mypy src/tech_doc_scanner/converter.py

# Ignore missing imports
mypy --ignore-missing-imports src/
```

## Useful Commands

```bash
# Run all checks
pytest && ruff check src/ tests/ && black --check src/ tests/ && mypy src/

# Clean build artifacts
rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache htmlcov/

# Regenerate lock file
uv lock

# Update dependencies
uv sync --upgrade
```

## Resources

- **Docling**: https://github.com/DS4SD/docling
- **Pix2Text**: https://github.com/breezedeus/Pix2Text
- **Click**: https://click.palletsprojects.com/
- **Pydantic**: https://docs.pydantic.dev/
- **Pytest**: https://docs.pytest.org/

## Questions?

- Check existing issues: [GitHub Issues](https://github.com/yourusername/tech-doc-scanner/issues)
- Read the docs: [Documentation](../README.md)
- Ask the community: [Discussions](https://github.com/yourusername/tech-doc-scanner/discussions)
