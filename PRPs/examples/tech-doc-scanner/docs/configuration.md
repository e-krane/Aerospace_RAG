# Configuration Reference

## Overview

Tech Doc Scanner uses a hierarchical configuration system with three main components:

1. **DoclingConfig**: PDF processing and OCR settings
2. **Pix2TextConfig**: Formula recognition settings
3. **OutputConfig**: Output format and path settings

## Configuration Methods

### 1. Direct Python API

```python
from tech_doc_scanner import DocumentConverterWrapper, DoclingConfig, Pix2TextConfig, OutputConfig

converter = DocumentConverterWrapper(
    pipeline_type="pix2text",
    docling_config=DoclingConfig(...),
    pix2text_config=Pix2TextConfig(...),
    output_config=OutputConfig(...),
)
```

### 2. YAML Configuration File

```python
from tech_doc_scanner.config import Config

# Load from YAML
config = Config.from_yaml("config.yaml")

# Create converter
converter = DocumentConverterWrapper(
    pipeline_type="pix2text",
    docling_config=config.docling,
    pix2text_config=config.pix2text,
    output_config=config.output,
)
```

### 3. CLI Flags

```bash
tech-doc-scanner convert input.pdf \
    --pipeline pix2text \
    --device cuda \
    --no-validation \
    --output-dir ./output
```

## DoclingConfig Options

Controls Docling PDF processing pipeline.

```python
DoclingConfig(
    do_ocr=True,                           # Enable OCR
    ocr_languages=["en"],                  # OCR languages (ISO codes)
    do_table_structure=True,               # Extract table structure
    table_mode="accurate",                 # or "fast"
    do_picture_classification=True,        # Classify images
    do_code_enrichment=True,               # Detect code blocks
    accelerator_device="auto",             # "auto", "cpu", "cuda", "mps"
    accelerator_threads=4,                 # CPU threads for acceleration
)
```

### OCR Languages

Supported language codes (examples):
- `en` - English
- `de` - German
- `fr` - French
- `es` - Spanish
- `zh` - Chinese
- Multiple: `["en", "de", "fr"]`

### Table Modes

- `accurate`: Better quality, slower (recommended)
- `fast`: Faster processing, may miss details

## Pix2TextConfig Options

Controls formula recognition with Pix2Text.

```python
Pix2TextConfig(
    enabled=True,                          # Enable Pix2Text processing
    device="cpu",                          # "cpu", "cuda", "mps"
    validate=True,                         # Enable LaTeX validation
    clean=True,                            # Enable LaTeX cleaning
    fallback_to_ocr=True,                  # OCR fallback on validation failure
    max_clean_iterations=3,                # Max cleaning iterations
    images_scale=2.6,                      # Formula image scale factor
    expansion_factor=0.1,                  # Context expansion around formulas
)
```

### Device Options

- `cpu`: CPU processing (slower, works everywhere)
- `cuda`: NVIDIA GPU (fastest, requires CUDA)
- `mps`: Apple Metal (M1/M2 Macs)

### Validation & Fallback

The validation system prevents KaTeX rendering errors:

1. **validate=True**: Checks LaTeX for common errors
   - Unbalanced braces
   - Incomplete commands
   - Missing environment closures
   - Infinite patterns

2. **clean=True**: Fixes common LaTeX issues
   - Balances braces
   - Fixes `\left`/`\right` pairs
   - Removes trailing operators
   - Cleans whitespace

3. **fallback_to_ocr=True**: Uses OCR if validation fails
   - Prevents crashes on malformed LaTeX
   - Ensures 100% conversion success
   - May have lower quality than MFR

### Formula Image Processing

- **images_scale**: Higher = better quality, slower
  - Range: 1.0 - 4.0
  - Default: 2.6 (good balance)
  - Recommended: 3.0+ for complex formulas

- **expansion_factor**: Context around formulas
  - Range: 0.0 - 0.3
  - Default: 0.1 (10% expansion)
  - Higher = more context, may include noise

## OutputConfig Options

Controls output files and formats.

```python
OutputConfig(
    formats=["md", "html"],                # Output formats
    base_dir="./output",                   # Output directory
)
```

### Supported Formats

- `md`: Markdown (default)
- `html`: HTML with embedded styles
- `json`: Docling JSON format
- `doctags`: Tagged document format

## Complete YAML Example

```yaml
docling:
  do_ocr: true
  ocr_languages:
    - en
    - de
  do_table_structure: true
  table_mode: accurate
  do_picture_classification: true
  do_code_enrichment: true
  accelerator_device: auto
  accelerator_threads: 4

pix2text:
  enabled: true
  device: cuda
  validate: true
  clean: true
  fallback_to_ocr: true
  max_clean_iterations: 3
  images_scale: 2.6
  expansion_factor: 0.1

output:
  formats:
    - md
    - html
  base_dir: ./output
```

Save as `config.yaml` and use:

```bash
tech-doc-scanner convert input.pdf --config config.yaml
```

## Presets

### Fast Conversion (No Formulas)

```python
DoclingConfig()  # defaults
Pix2TextConfig(enabled=False)
```

### Best Quality (GPU)

```python
DoclingConfig(
    table_mode="accurate",
    do_picture_classification=True,
    accelerator_device="cuda",
)
Pix2TextConfig(
    device="cuda",
    validate=True,
    clean=True,
    images_scale=3.0,
)
```

### CPU-Only Production

```python
DoclingConfig(accelerator_device="cpu")
Pix2TextConfig(
    device="cpu",
    validate=True,
    fallback_to_ocr=True,
)
```

### Multi-Language OCR

```python
DoclingConfig(
    do_ocr=True,
    ocr_languages=["en", "de", "fr", "es"],
)
```

## Environment Variables

Currently not supported, but you can set defaults in a shared config file.

## Validation

Check if your YAML config is valid:

```bash
tech-doc-scanner config-validate config.yaml
```

Generate a template config:

```bash
tech-doc-scanner config-generate -o template.yaml
```

## Tips

1. **For best formula quality**: `validate=True`, `clean=True`, `images_scale=3.0`
2. **For speed**: `enabled=False` (disable Pix2Text)
3. **For GPU**: Ensure CUDA/MPS is available, set `device`
4. **For debugging**: Enable verbose mode: `--verbose`
5. **For batch processing**: Reuse config across files

## Next Steps

- Try [Examples](../examples/README.md)
- Read [API Documentation](api.md)
- See [Development Guide](development.md)
