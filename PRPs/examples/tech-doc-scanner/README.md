# Technical Document Scanner

> High-quality PDF to Markdown conversion for technical documents with state-of-the-art equation recognition

**Tech Doc Scanner** is a production-ready application that converts technical PDFs to Markdown with the highest possible quality equation handling. Built on [Docling](https://github.com/docling-project/docling) for structure detection and [Pix2Text](https://github.com/breezedeus/Pix2Text) for mathematical formula recognition.

## Features

- **Equation Recognition**: State-of-the-art MFR 1.5 model from Pix2Text
- **Structure Detection**: Tables, layouts, and document structure via Docling
- **LaTeX Validation**: Automatic cleaning and validation with OCR fallback
- **GPU Acceleration**: CUDA/MPS support for fast processing
- **Batch Processing**: Process multiple PDFs with model caching
- **Unified CLI**: Simple command-line interface
- **Python API**: Programmatic access for custom workflows

## Quick Start

### Installation

\`\`\`bash
# Clone the repository
git clone <repository-url>
cd docling_testing

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
\`\`\`

### Basic Usage

\`\`\`bash
# Convert a single PDF
tech-doc-scanner convert input.pdf --output-dir ./output

# Batch process multiple PDFs
tech-doc-scanner batch ./pdfs/*.pdf --output-dir ./outputs

# Use configuration file
tech-doc-scanner convert input.pdf --config config.yaml
\`\`\`

### Python API

\`\`\`python
from tech_doc_scanner import DocumentConverterWrapper, DoclingConfig, Pix2TextConfig

# Create configuration
docling_config = DoclingConfig(
    accelerator_device="cuda",
    ocr_languages=["en"],
)

pix2text_config = Pix2TextConfig(
    enabled=True,
    device="cuda",
    validate=True,
    fallback=True,
)

# Create converter
converter = DocumentConverterWrapper(
    pipeline_type="pix2text",
    docling_config=docling_config,
    pix2text_config=pix2text_config,
)

# Convert document
result = converter.convert_file(
    input_path="input.pdf",
    output_dir="./output",
)

# Access statistics
print(f"Formulas processed: {result.stats.formulas_total}")
print(f"Success rate: {result.stats.success_rate:.1f}%")
\`\`\`

## Architecture

Tech Doc Scanner consolidates the best approaches from extensive experimentation with PDF conversion:

1. **Docling** handles structure detection, table extraction, and OCR
2. **Pix2Text** recognizes mathematical formulas with MFR 1.5 (SOTA)
3. **LaTeX Processing** cleans and validates formulas before output
4. **Validation & Fallback** ensures 100% valid output or falls back to OCR

This combination provides the highest quality conversion for technical documents with complex equations.

## Project Origin

This project evolved from extensive experimentation with different PDF conversion approaches:

- Started with 9+ experimental scripts testing various pipelines
- Discovered optimal combination: Docling structure + Pix2Text formulas
- Added LaTeX validation to prevent rendering errors
- Refactored into production-ready modular architecture

See \`examples/archive/\` for the experimental scripts that led to this solution.

## Documentation

- [Installation Guide](docs/installation.md) - Setup and dependencies
- [Configuration Reference](docs/configuration.md) - All configuration options
- [API Documentation](docs/api.md) - Python API reference
- [Development Guide](docs/development.md) - Contributing and development

## Requirements

- Python 3.13+
- CUDA-capable GPU (optional, for acceleration)
- ~2GB RAM for Pix2Text models

## Performance

- **Standard Pipeline**: ~20 seconds per document
- **Pix2Text Pipeline**: ~5 minutes per document (formula-heavy)
- **Batch Processing**: Faster due to model caching

## Contributing

Contributions are welcome! Please see [DEVELOPMENT.md](docs/development.md) for guidelines.

## License

[License details to be added]

## Acknowledgments

- [Docling](https://github.com/docling-project/docling) - PDF structure detection
- [Pix2Text](https://github.com/breezedeus/Pix2Text) - Mathematical formula recognition
- Original experimentation codebase from docling-testing project
