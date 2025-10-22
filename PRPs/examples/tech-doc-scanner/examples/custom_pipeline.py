"""
Custom pipeline configuration example.

This example shows how to customize pipeline settings for specific needs,
such as OCR languages, GPU acceleration, and formula processing options.
"""

from pathlib import Path
from tech_doc_scanner import DocumentConverterWrapper, DoclingConfig, Pix2TextConfig, OutputConfig
from tech_doc_scanner.config import Config


def example_gpu_acceleration():
    """Example: Using GPU acceleration for faster processing."""
    print("\n=== GPU Acceleration Example ===\n")
    
    converter = DocumentConverterWrapper(
        pipeline_type="pix2text",
        docling_config=DoclingConfig(),
        pix2text_config=Pix2TextConfig(
            device="cuda",  # Use GPU (requires CUDA)
            validate=True,
            fallback_to_ocr=True,
        ),
        output_config=OutputConfig(base_dir=Path("./output/gpu_example")),
    )
    
    print("Converter configured with CUDA GPU acceleration")
    print("Note: Requires CUDA-capable GPU and drivers")


def example_custom_ocr():
    """Example: Configuring OCR for multiple languages."""
    print("\n=== Custom OCR Example ===\n")
    
    converter = DocumentConverterWrapper(
        pipeline_type="pix2text",
        docling_config=DoclingConfig(
            do_ocr=True,
            ocr_languages=["en", "de", "fr"],  # Multiple languages
        ),
        pix2text_config=Pix2TextConfig(device="cpu"),
        output_config=OutputConfig(base_dir=Path("./output/multilang_example")),
    )
    
    print("Converter configured for English, German, and French OCR")


def example_config_file():
    """Example: Using YAML configuration file."""
    print("\n=== Configuration File Example ===\n")
    
    # Create a custom configuration
    config = Config(
        docling=DoclingConfig(
            do_ocr=True,
            ocr_languages=["en"],
            do_table_structure=True,
        ),
        pix2text=Pix2TextConfig(
            device="cpu",
            validate=True,
            clean=True,
            fallback_to_ocr=True,
        ),
        output=OutputConfig(
            formats=["md", "html", "json"],
            base_dir=Path("./output/custom"),
        ),
    )
    
    # Save to YAML
    config_path = Path("custom_config.yaml")
    config.to_yaml(config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Load and use
    loaded_config = Config.from_yaml(config_path)
    converter = DocumentConverterWrapper(
        pipeline_type="pix2text",
        docling_config=loaded_config.docling,
        pix2text_config=loaded_config.pix2text,
        output_config=loaded_config.output,
    )
    
    print(f"Configuration loaded from: {config_path}")
    print(f"Output formats: {loaded_config.output.formats}")


def example_formula_tuning():
    """Example: Fine-tuning formula recognition settings."""
    print("\n=== Formula Recognition Tuning Example ===\n")
    
    converter = DocumentConverterWrapper(
        pipeline_type="pix2text",
        docling_config=DoclingConfig(),
        pix2text_config=Pix2TextConfig(
            device="cpu",
            validate=True,          # Enable LaTeX validation
            clean=True,             # Enable LaTeX cleaning
            fallback_to_ocr=True,   # Use OCR if validation fails
            max_clean_iterations=5,  # More cleaning iterations
        ),
        output_config=OutputConfig(base_dir=Path("./output/tuned_formulas")),
    )
    
    print("Converter configured with enhanced formula processing:")
    print("  ✓ LaTeX validation enabled")
    print("  ✓ LaTeX cleaning enabled (5 iterations)")
    print("  ✓ OCR fallback on validation failure")


def main():
    """Run all customization examples."""
    print("Tech Doc Scanner - Custom Pipeline Configuration Examples")
    print("=" * 60)
    
    example_gpu_acceleration()
    example_custom_ocr()
    example_config_file()
    example_formula_tuning()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("\nFor actual conversions, use these configurations with:")
    print("  result = converter.convert_file(pdf_path, output_dir)")


if __name__ == "__main__":
    main()
