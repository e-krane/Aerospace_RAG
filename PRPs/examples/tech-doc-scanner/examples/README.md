# Tech Doc Scanner Examples

This directory contains practical examples demonstrating how to use the tech-doc-scanner library.

## Quick Start Examples

### 1. Basic Conversion (`basic_conversion.py`)

The simplest way to convert a PDF to Markdown:

```python
from tech_doc_scanner import DocumentConverterWrapper, DoclingConfig, Pix2TextConfig, OutputConfig

converter = DocumentConverterWrapper(
    pipeline_type="pix2text",
    docling_config=DoclingConfig(),
    pix2text_config=Pix2TextConfig(device="cpu"),
    output_config=OutputConfig(base_dir="./output"),
)

result = converter.convert_file("document.pdf", "./output")
```

**Run**: `python examples/basic_conversion.py`

### 2. Batch Processing (`batch_processing.py`)

Convert multiple PDFs with consolidated reporting:

- Processes all PDFs matching a glob pattern
- Collects statistics across all conversions
- Generates HTML report with summary

**Run**: `python examples/batch_processing.py`

### 3. Pipeline Comparison (`compare_pipelines.py`)

Compare standard vs pix2text pipelines:

- **Standard**: Faster, good structure detection, no formula recognition
- **Pix2Text**: Slower, best quality formula recognition with validation

**Run**: `python examples/compare_pipelines.py`

### 4. Custom Configuration (`custom_pipeline.py`)

Advanced configuration examples:

- GPU acceleration (CUDA)
- Multi-language OCR
- YAML configuration files
- Formula recognition tuning

**Run**: `python examples/custom_pipeline.py`

## Using the CLI

For quick conversions without writing code, use the command-line interface:

```bash
# Basic conversion
tech-doc-scanner convert input.pdf

# With GPU acceleration
tech-doc-scanner convert input.pdf --device cuda

# Batch processing with report
tech-doc-scanner batch "pdfs/*.pdf" --report

# Using config file
tech-doc-scanner config-generate -o my-config.yaml
# Edit my-config.yaml...
tech-doc-scanner convert input.pdf --config my-config.yaml
```

## Pipeline Types

### Standard Pipeline
- **Speed**: Fast (~20s for typical document)
- **Use case**: Documents without equations, quick conversions
- **Features**: Structure detection, table extraction, OCR

### Pix2Text Pipeline (Recommended)
- **Speed**: Moderate (~5 min for technical document)
- **Use case**: Technical documents with mathematical formulas
- **Features**: Everything in Standard + state-of-the-art formula recognition
- **Formula Processing**:
  - Pix2Text MFR 1.5 model (SOTA)
  - LaTeX cleaning (6 methods)
  - LaTeX validation (7 checks)
  - Automatic OCR fallback

## Configuration Options

### Docling Configuration

```python
DoclingConfig(
    do_ocr=True,                    # Enable OCR
    ocr_languages=["en"],           # OCR languages
    do_table_structure=True,        # Extract tables
    do_picture_classification=True, # Classify images
)
```

### Pix2Text Configuration

```python
Pix2TextConfig(
    device="cpu",              # or "cuda" for GPU
    validate=True,             # Enable LaTeX validation
    clean=True,                # Enable LaTeX cleaning
    fallback_to_ocr=True,      # OCR fallback on validation failure
    max_clean_iterations=3,    # Cleaning iterations
)
```

### Output Configuration

```python
OutputConfig(
    formats=["md", "html"],    # Output formats
    base_dir="./output",       # Output directory
)
```

## Tips

1. **For best formula quality**: Use pix2text pipeline with validation enabled
2. **For speed**: Use standard pipeline or disable validation
3. **For GPU**: Ensure CUDA is installed, use `device="cuda"`
4. **For multiple documents**: Use batch processing to reuse model cache
5. **For custom needs**: Save your config to YAML and reuse it

## Archive

The `archive/` directory contains the original experimental scripts that were used during development. These are kept for reference but are **deprecated** - use the new library instead!

- All `docling_testing_*.py` scripts have been superseded by the unified library
- See `archive/DEPRECATED.md` for migration guide

## Support

For more information:
- Package documentation: `README.md` in project root
- API reference: Check source code docstrings
- CLI help: `tech-doc-scanner --help`
