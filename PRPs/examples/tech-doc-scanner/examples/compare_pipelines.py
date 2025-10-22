"""
Pipeline comparison example.

This example compares standard vs pix2text pipelines to show
the difference in formula recognition quality.
"""

from pathlib import Path
from tech_doc_scanner import DocumentConverterWrapper, DoclingConfig, Pix2TextConfig, OutputConfig


def convert_with_pipeline(pdf_path: Path, pipeline_type: str, output_subdir: str) -> None:
    """Convert PDF with specified pipeline type."""
    output_dir = Path(f"./output/compare_{output_subdir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create appropriate config based on pipeline type
    pix2text_config = Pix2TextConfig(device="cpu") if pipeline_type == "pix2text" else Pix2TextConfig()
    
    converter = DocumentConverterWrapper(
        pipeline_type=pipeline_type,
        docling_config=DoclingConfig(),
        pix2text_config=pix2text_config,
        output_config=OutputConfig(base_dir=output_dir),
    )
    
    print(f"\n{'='*60}")
    print(f"Pipeline: {pipeline_type.upper()}")
    print(f"{'='*60}")
    
    result = converter.convert_file(pdf_path, output_dir)
    
    if result.success:
        print(f"✓ Conversion successful in {result.elapsed_time:.1f}s")
        print(f"Output: {output_dir / pdf_path.stem}.md")
        
        if result.stats and pipeline_type == "pix2text":
            print(f"\nFormula Statistics:")
            print(f"  Total formulas: {result.stats.total}")
            print(f"  Recognized: {result.stats.recognized}")
            print(f"  Cleaned: {result.stats.cleaned}")
            print(f"  Validation failed: {result.stats.validation_failed}")
            print(f"  Fallback used: {result.stats.fallback_used}")
            print(f"  Success rate: {result.stats.success_rate:.1f}%")
        elif pipeline_type == "standard":
            print("\n(Standard pipeline does not include formula recognition)")
    else:
        print(f"✗ Conversion failed: {result.error}")


def main():
    """Compare standard and pix2text pipelines."""
    pdf_path = Path("Bruhn_Crippling_short.pdf")
    
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return
    
    print(f"Comparing pipelines on: {pdf_path}")
    
    # Convert with standard pipeline (faster, no formula recognition)
    convert_with_pipeline(pdf_path, "standard", "standard")
    
    # Convert with pix2text pipeline (slower, best formula quality)
    convert_with_pipeline(pdf_path, "pix2text", "pix2text")
    
    print(f"\n{'='*60}")
    print("Comparison complete!")
    print("\nRecommendations:")
    print("  • Standard pipeline: Fast, good for documents without equations")
    print("  • Pix2Text pipeline: Best for technical documents with formulas")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
