"""
Configuration module for Tech Doc Scanner.

Provides dataclass-based configuration with validation, defaults, and YAML serialization.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions, TableFormerMode


class DeviceType(str, Enum):
    """Supported device types for acceleration."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class TableMode(str, Enum):
    """Table extraction modes."""

    FAST = "fast"
    ACCURATE = "accurate"


class ImageMode(str, Enum):
    """Image reference modes for exports."""

    EMBEDDED = "embedded"
    REFERENCED = "referenced"
    PLACEHOLDER = "placeholder"


@dataclass
class DoclingConfig:
    """
    Configuration for Docling PDF pipeline.

    Attributes:
        do_ocr: Enable OCR for text extraction
        ocr_languages: List of language codes for OCR (e.g., ["en", "de"])
        do_table_structure: Enable table structure extraction
        table_mode: Table extraction mode (fast or accurate)
        do_cell_matching: Enable cell matching in tables
        generate_picture_images: Generate images from PDF
        images_scale: Scale factor for images
        do_picture_classification: Enable picture classification
        do_code_enrichment: Enable code block enrichment
        accelerator_device: Device for acceleration (auto, cpu, cuda, mps)
        accelerator_threads: Number of threads for processing
    """

    # OCR settings
    do_ocr: bool = True
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])

    # Table extraction
    do_table_structure: bool = True
    table_mode: TableMode = TableMode.ACCURATE
    do_cell_matching: bool = True

    # Image processing
    generate_picture_images: bool = True
    images_scale: float = 2.0

    # Other enrichments
    do_picture_classification: bool = True
    do_code_enrichment: bool = True

    # Acceleration
    accelerator_device: DeviceType = DeviceType.AUTO
    accelerator_threads: int = 4

    def to_pipeline_options(self) -> PdfPipelineOptions:
        """
        Convert to Docling's PdfPipelineOptions.

        Returns:
            PdfPipelineOptions configured with this config
        """
        options = PdfPipelineOptions()

        # OCR
        options.do_ocr = self.do_ocr
        if self.do_ocr:
            options.ocr_options = EasyOcrOptions(lang=self.ocr_languages)

        # Table structure
        options.do_table_structure = self.do_table_structure
        if self.do_table_structure:
            options.table_structure_options.mode = (
                TableFormerMode.ACCURATE
                if self.table_mode == TableMode.ACCURATE
                else TableFormerMode.FAST
            )
            options.table_structure_options.do_cell_matching = self.do_cell_matching

        # Images
        options.generate_picture_images = self.generate_picture_images
        options.images_scale = self.images_scale

        # Other enrichments
        options.do_picture_classification = self.do_picture_classification
        options.do_code_enrichment = self.do_code_enrichment

        # Accelerator
        device_map = {
            DeviceType.AUTO: AcceleratorDevice.AUTO,
            DeviceType.CPU: AcceleratorDevice.CPU,
            DeviceType.CUDA: AcceleratorDevice.CUDA,
            DeviceType.MPS: AcceleratorDevice.MPS,
        }
        options.accelerator_options = AcceleratorOptions(
            num_threads=self.accelerator_threads, device=device_map[self.accelerator_device]
        )

        return options

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DoclingConfig":
        """
        Create config from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            DoclingConfig instance
        """
        # Convert string enums
        if "table_mode" in data and isinstance(data["table_mode"], str):
            data["table_mode"] = TableMode(data["table_mode"])
        if "accelerator_device" in data and isinstance(data["accelerator_device"], str):
            data["accelerator_device"] = DeviceType(data["accelerator_device"])

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "do_ocr": self.do_ocr,
            "ocr_languages": self.ocr_languages,
            "do_table_structure": self.do_table_structure,
            "table_mode": self.table_mode.value,
            "do_cell_matching": self.do_cell_matching,
            "generate_picture_images": self.generate_picture_images,
            "images_scale": self.images_scale,
            "do_picture_classification": self.do_picture_classification,
            "do_code_enrichment": self.do_code_enrichment,
            "accelerator_device": self.accelerator_device.value,
            "accelerator_threads": self.accelerator_threads,
        }


@dataclass
class Pix2TextConfig:
    """
    Configuration for Pix2Text formula recognition.

    Attributes:
        enabled: Enable Pix2Text formula recognition
        device: Device for Pix2Text (cpu or cuda)
        validate: Enable LaTeX validation
        clean: Enable LaTeX cleaning
        fallback_to_ocr: Fallback to OCR when validation fails
        max_clean_iterations: Maximum cleaning iterations
        images_scale: Scale factor for formula images
        expansion_factor: Context expansion around formulas
    """

    enabled: bool = True
    device: str = "cuda"
    validate: bool = True
    clean: bool = True
    fallback_to_ocr: bool = True
    max_clean_iterations: int = 3
    images_scale: float = 2.6
    expansion_factor: float = 0.1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pix2TextConfig":
        """Create config from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "device": self.device,
            "validate": self.validate,
            "clean": self.clean,
            "fallback_to_ocr": self.fallback_to_ocr,
            "max_clean_iterations": self.max_clean_iterations,
            "images_scale": self.images_scale,
            "expansion_factor": self.expansion_factor,
        }


@dataclass
class OutputConfig:
    """
    Configuration for output formats and paths.

    Attributes:
        formats: List of output formats (markdown, html, json)
        image_mode: How to handle images in exports
        base_dir: Base directory for outputs
        create_subdirs: Create subdirectories per conversion
    """

    formats: List[str] = field(default_factory=lambda: ["markdown", "html"])
    image_mode: ImageMode = ImageMode.REFERENCED
    base_dir: Path = field(default_factory=lambda: Path("./output"))
    create_subdirs: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        """Create config from dictionary."""
        if "image_mode" in data and isinstance(data["image_mode"], str):
            data["image_mode"] = ImageMode(data["image_mode"])
        if "base_dir" in data:
            data["base_dir"] = Path(data["base_dir"])
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "formats": self.formats,
            "image_mode": self.image_mode.value,
            "base_dir": str(self.base_dir),
            "create_subdirs": self.create_subdirs,
        }


@dataclass
class Config:
    """
    Complete application configuration.

    Attributes:
        docling: Docling pipeline configuration
        pix2text: Pix2Text formula recognition configuration
        output: Output format and path configuration
    """

    docling: DoclingConfig = field(default_factory=DoclingConfig)
    pix2text: Pix2TextConfig = field(default_factory=Pix2TextConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty config file: {path}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        Create config from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Config instance
        """
        docling_data = data.get("docling", {})
        pix2text_data = data.get("pix2text", {})
        output_data = data.get("output", {})

        return cls(
            docling=DoclingConfig.from_dict(docling_data),
            pix2text=Pix2TextConfig.from_dict(pix2text_data),
            output=OutputConfig.from_dict(output_data),
        )

    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration file
        """
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "docling": self.docling.to_dict(),
            "pix2text": self.pix2text.to_dict(),
            "output": self.output.to_dict(),
        }


# Default configuration instance
DEFAULT_CONFIG = Config()
