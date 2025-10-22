"""
Basic PDF to Markdown conversion example.

This example shows the simplest way to convert a PDF to Markdown
using tech-doc-scanner with default settings.
"""

from pathlib import Path
from tech_doc_scanner import DocumentConverterWrapper, DoclingConfig, Pix2TextConfig, OutputConfig


def main():
    """Convert a single PDF to Markdown with default settings."""
    # Input and output paths
    input_pdf = Path("Bruhn_Crippling_short.pdf")
    output_dir = Path("./output/basic_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create converter with pix2text pipeline (best quality)
    converter = DocumentConverterWrapper(
        pipeline_type="pix2text",
        docling_config=DoclingConfig(),
        pix2text_config=Pix2TextConfig(device="cpu"),  # Use "cuda" for GPU
        output_config=OutputConfig(base_dir=output_dir),
    )
    
    # Convert the PDF
    print(f"Converting: {input_pdf}")
    result = converter.convert_file(input_pdf, output_dir)
    
    # Display results
    if result.success:
        print(f"\n✓ Conversion successful in {result.elapsed_time:.1f}s")
        print(f"\nOutput files:")
        for file in result.output_files:
            print(f"  • {file}")
        
        # Display formula stats if available
        if result.stats:
            print(f"\nFormula Recognition:")
            print(f"  Total: {result.stats.total}")
            print(f"  Recognized: {result.stats.recognized}")
            print(f"  Success rate: {result.stats.success_rate:.1f}%")
    else:
        print(f"\n✗ Conversion failed: {result.error}")


if __name__ == "__main__":
    main()
