# API Documentation

## Quick Start

```python
from tech_doc_scanner import DocumentConverterWrapper, DoclingConfig, Pix2TextConfig, OutputConfig

# Create converter
converter = DocumentConverterWrapper(
    pipeline_type="pix2text",
    docling_config=DoclingConfig(),
    pix2text_config=Pix2TextConfig(device="cpu"),
    output_config=OutputConfig(base_dir="./output"),
)

# Convert a PDF
result = converter.convert_file("document.pdf", "./output")

# Check results
if result.success:
    print(f"Converted in {result.elapsed_time:.1f}s")
    for file in result.output_files:
        print(f"  Output: {file}")
else:
    print(f"Error: {result.error}")
```

## Core Classes

### DocumentConverterWrapper

Main conversion interface.

```python
class DocumentConverterWrapper:
    def __init__(
        self,
        pipeline_type: str,             # "standard" or "pix2text"
        docling_config: DoclingConfig,
        pix2text_config: Pix2TextConfig,
        output_config: OutputConfig,
    ):
        """Initialize converter with configuration."""
        ...

    def convert_file(
        self,
        input_path: Path,
        output_dir: Path,
    ) -> ConversionResult:
        """
        Convert a single PDF file.
        
        Args:
            input_path: Path to PDF file
            output_dir: Output directory
            
        Returns:
            ConversionResult with success status, files, stats, errors
        """
        ...
```

### ConversionResult

Result of document conversion.

```python
@dataclass
class ConversionResult:
    input_path: Path                     # Input PDF path
    output_dir: Path                     # Output directory
    success: bool                        # Whether conversion succeeded
    elapsed_time: float                  # Time in seconds
    stats: Optional[EnrichmentStats]     # Formula stats (if applicable)
    error: Optional[str]                 # Error message if failed
    output_files: List[Path]             # Generated output files
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        ...
```

### Configuration Classes

#### DoclingConfig

```python
@dataclass
class DoclingConfig:
    do_ocr: bool = True
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    do_table_structure: bool = True
    table_mode: str = "accurate"         # or "fast"
    do_picture_classification: bool = True
    do_code_enrichment: bool = True
    accelerator_device: str = "auto"     # "auto", "cpu", "cuda", "mps"
    accelerator_threads: int = 4
```

#### Pix2TextConfig

```python
@dataclass
class Pix2TextConfig:
    enabled: bool = True
    device: str = "cuda"                 # "cpu", "cuda", "mps"
    validate: bool = True
    clean: bool = True
    fallback_to_ocr: bool = True
    max_clean_iterations: int = 3
    images_scale: float = 2.6
    expansion_factor: float = 0.1
```

#### OutputConfig

```python
@dataclass
class OutputConfig:
    formats: List[str] = field(default_factory=lambda: ["md", "html"])
    base_dir: Path = Path("./output")
```

### Statistics Classes

#### EnrichmentStats

```python
@dataclass
class EnrichmentStats:
    total: int = 0                       # Total formulas detected
    recognized: int = 0                  # Successfully recognized
    cleaned: int = 0                     # LaTeX cleaning applied
    validation_failed: int = 0           # Validation failures
    fallback_used: int = 0               # OCR fallback used
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        ...
    
    def print_summary(self) -> None:
        """Print formatted summary to console."""
        ...
```

#### BatchStatistics

```python
class BatchStatistics:
    def __init__(self):
        """Initialize batch statistics collector."""
        ...
    
    def add_conversion(self, stats: ConversionStatistics) -> None:
        """Add conversion statistics to batch."""
        ...
    
    def print_summary(self) -> None:
        """Print batch summary to console."""
        ...
    
    def generate_html_report(self, output_path: Path) -> None:
        """Generate HTML report of batch statistics."""
        ...
```

## Pipeline Factory

Create pipelines programmatically.

```python
from tech_doc_scanner.pipelines import PipelineFactory

# Standard pipeline (fast)
pipeline = PipelineFactory.create_standard_pipeline(docling_config)

# Pix2Text pipeline (best quality)
pipeline = PipelineFactory.create_pix2text_pipeline(
    docling_config, 
    pix2text_config
)

# By string
pipeline = PipelineFactory.create_pipeline(
    "pix2text",
    docling_config,
    pix2text_config
)
```

## LaTeX Processing

Low-level LaTeX cleaning and validation.

```python
from tech_doc_scanner.latex import LaTeXCleaner, LaTeXValidator

# Clean LaTeX
cleaner = LaTeXCleaner()
cleaned = cleaner.clean(raw_latex)

# Validate LaTeX
validator = LaTeXValidator()
is_valid, error = validator.validate(latex_string)
```

## Configuration Management

Load/save YAML configurations.

```python
from tech_doc_scanner.config import Config

# Create config
config = Config(
    docling=DoclingConfig(...),
    pix2text=Pix2TextConfig(...),
    output=OutputConfig(...),
)

# Save to YAML
config.to_yaml("config.yaml")

# Load from YAML
config = Config.from_yaml("config.yaml")
```

## Examples

### Basic Conversion

```python
from pathlib import Path
from tech_doc_scanner import DocumentConverterWrapper, DoclingConfig, Pix2TextConfig, OutputConfig

converter = DocumentConverterWrapper(
    pipeline_type="pix2text",
    docling_config=DoclingConfig(),
    pix2text_config=Pix2TextConfig(device="cpu"),
    output_config=OutputConfig(base_dir=Path("./output")),
)

result = converter.convert_file(Path("document.pdf"), Path("./output"))
```

### With Custom Configuration

```python
converter = DocumentConverterWrapper(
    pipeline_type="pix2text",
    docling_config=DoclingConfig(
        do_ocr=True,
        ocr_languages=["en", "de"],
        table_mode="accurate",
    ),
    pix2text_config=Pix2TextConfig(
        device="cuda",
        validate=True,
        clean=True,
        images_scale=3.0,
    ),
    output_config=OutputConfig(
        formats=["md", "html", "json"],
        base_dir=Path("./output"),
    ),
)
```

### Batch Processing

```python
from tech_doc_scanner.stats import BatchStatistics, ConversionStatistics
import glob

batch_stats = BatchStatistics()

for pdf_file in glob.glob("*.pdf"):
    result = converter.convert_file(Path(pdf_file), output_dir)
    
    stats = ConversionStatistics(
        input_file=pdf_file,
        success=result.success,
        elapsed_time=result.elapsed_time,
        output_formats=[f.suffix[1:] for f in result.output_files],
        formula_stats=result.stats.to_dict() if result.stats else None,
        error=result.error,
    )
    batch_stats.add_conversion(stats)

batch_stats.print_summary()
batch_stats.generate_html_report(Path("report.html"))
```

## Error Handling

All conversion errors are captured in ConversionResult:

```python
result = converter.convert_file(pdf_path, output_dir)

if not result.success:
    print(f"Conversion failed: {result.error}")
    # Handle error appropriately
else:
    # Process successful result
    pass
```

## Type Hints

The entire API is fully type-hinted for IDE support:

```python
from typing import List, Optional
from pathlib import Path

def process_documents(files: List[Path]) -> List[ConversionResult]:
    results: List[ConversionResult] = []
    for file in files:
        result = converter.convert_file(file, output_dir)
        results.append(result)
    return results
```

## See Also

- [Configuration Reference](configuration.md)
- [Examples](../examples/README.md)
- [Development Guide](development.md)
