# Installation Guide

## Prerequisites

- Python 3.9 or higher
- pip or uv package manager
- (Optional) CUDA-capable GPU for acceleration

## Quick Install

Using uv (recommended):

```bash
uv sync
```

Using pip:

```bash
pip install -e .
```

## Verify Installation

```bash
# Check CLI is available
tech-doc-scanner --version

# Or
tds --version

# Run help
tech-doc-scanner --help
```

## GPU Support (Optional)

For CUDA GPU acceleration:

1. Install CUDA Toolkit (11.8 or later)
2. Install cuDNN
3. Verify CUDA is working:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

Then use `--device cuda` flag with CLI or `device="cuda"` in Python API.

## Development Installation

For development with testing tools:

```bash
# Clone repository
git clone https://github.com/yourusername/tech-doc-scanner.git
cd tech-doc-scanner

# Install with dev dependencies
uv sync

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/tech_doc_scanner --cov-report=html
```

## Dependencies

Core dependencies:
- **docling**: PDF structure detection and conversion
- **pix2text**: Mathematical formula recognition (MFR 1.5)
- **click**: CLI framework
- **rich**: Terminal UI components
- **pydantic**: Configuration validation
- **pyyaml**: YAML config files
- **pillow**: Image processing

Development dependencies:
- pytest, pytest-cov, pytest-mock
- black, ruff, mypy

## Troubleshooting

### Command not found

If `tech-doc-scanner` command isn't found after installation:

```bash
# Using uv
uv run tech-doc-scanner --help

# Or activate virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
tech-doc-scanner --help
```

### CUDA out of memory

If you encounter CUDA OOM errors:
- Use `--device cpu` instead
- Process documents in smaller batches
- Reduce image quality settings

### Model download issues

Models are downloaded automatically on first use to:
- `~/.pix2text/` - Pix2Text models
- `~/.cnocr/` - OCR models

If downloads fail, check your internet connection and try again.

## Next Steps

- See [Configuration Guide](configuration.md) for settings
- Check [Examples](../examples/README.md) for usage patterns
- Read [API Documentation](api.md) for Python API
