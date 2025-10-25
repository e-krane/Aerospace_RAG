"""
Configuration management for Aerospace RAG system.

Loads YAML configuration files and provides type-safe access to settings.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass
import yaml

from loguru import logger


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    provider: str
    model: str
    dimensions: int
    matryoshka: bool
    reduced_dimensions: int
    batch_size: int
    normalize: bool
    instruction: str


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int
    retry_delay: int
    fallbacks: list


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    provider: str
    model: str
    thresholds: Dict[str, float]


@dataclass
class VRAMConfig:
    """VRAM usage configuration."""
    total: float
    embeddings: float
    llm: float
    peak: float
    simultaneous: float
    headroom: float
    mode: str


@dataclass
class ModelConfig:
    """Complete model configuration."""
    embeddings: EmbeddingConfig
    llm: LLMConfig
    evaluation: EvaluationConfig
    vram: VRAMConfig


@dataclass
class VectorDBConfig:
    """Vector database configuration."""
    type: str
    host: str
    port: int
    collection_name: str
    quantization: Dict[str, Any]
    vector_size: int
    distance: str
    on_disk_payload: bool
    wal_config: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    hybrid: Dict[str, Any]
    reranking: Dict[str, Any]
    max_results: int


@dataclass
class ChunkingConfig:
    """Chunking configuration."""
    type: str
    chunk_size: int
    overlap: int
    similarity_threshold: float
    min_chunk_size: int
    max_chunk_size: int
    preserve_equations: bool
    equation_boundary: bool


@dataclass
class SystemConfig:
    """Complete system configuration."""
    vector_db: VectorDBConfig
    retrieval: RetrievalConfig
    chunking: ChunkingConfig
    parsing: Optional[Dict[str, Any]] = None
    caching: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    targets: Optional[Dict[str, float]] = None
    paths: Optional[Dict[str, str]] = None


class ConfigLoader:
    """
    Configuration loader for Aerospace RAG system.

    Usage:
        config = ConfigLoader()
        model_config = config.models
        system_config = config.system

        # Access specific settings
        embedder_model = config.models.embeddings.model
        chunk_size = config.system.chunking.chunk_size
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        models_file: str = "models.yaml",
        system_file: str = "system.yaml",
    ):
        """
        Initialize configuration loader.

        Args:
            config_dir: Directory containing config files (defaults to project root/config)
            models_file: Name of models configuration file
            system_file: Name of system configuration file
        """
        if config_dir is None:
            # Default to project_root/config
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)
        self.models_file = self.config_dir / models_file
        self.system_file = self.config_dir / system_file

        # Load configurations
        self._models_raw = self._load_yaml(self.models_file)
        self._system_raw = self._load_yaml(self.system_file)

        # Parse into dataclasses
        self.models = self._parse_models(self._models_raw)
        self.system = self._parse_system(self._system_raw)

        logger.info(f"Configuration loaded from {self.config_dir}")

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r") as f:
            config = yaml.safe_load(f)

        logger.debug(f"Loaded configuration from {file_path}")
        return config

    def _parse_models(self, raw: Dict[str, Any]) -> ModelConfig:
        """Parse models configuration into dataclass."""
        return ModelConfig(
            embeddings=EmbeddingConfig(**raw["embeddings"]),
            llm=LLMConfig(**raw["llm"]),
            evaluation=EvaluationConfig(**raw["evaluation"]),
            vram=VRAMConfig(**raw["vram"]),
        )

    def _parse_system(self, raw: Dict[str, Any]) -> SystemConfig:
        """Parse system configuration into dataclass."""
        return SystemConfig(
            vector_db=VectorDBConfig(**raw["vector_db"]),
            retrieval=RetrievalConfig(**raw["retrieval"]),
            chunking=ChunkingConfig(**raw["chunking"]),
            parsing=raw.get("parsing"),
            caching=raw.get("caching"),
            monitoring=raw.get("monitoring"),
            logging=raw.get("logging"),
            targets=raw.get("targets"),
            paths=raw.get("paths"),
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., "models.embeddings.model")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        parts = key.split(".")

        # Determine which config to search
        if parts[0] == "models":
            obj = self.models
            parts = parts[1:]  # Remove "models" prefix
        elif parts[0] == "system":
            obj = self.system
            parts = parts[1:]  # Remove "system" prefix
        else:
            return default

        # Navigate through object
        for part in parts:
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return default

        return obj

    def reload(self):
        """Reload configuration from disk."""
        self._models_raw = self._load_yaml(self.models_file)
        self._system_raw = self._load_yaml(self.system_file)
        self.models = self._parse_models(self._models_raw)
        self.system = self._parse_system(self._system_raw)
        logger.info("Configuration reloaded")


# Global configuration instance
_config: Optional[ConfigLoader] = None


def get_config(reload: bool = False) -> ConfigLoader:
    """
    Get global configuration instance (singleton).

    Args:
        reload: Force reload configuration from disk

    Returns:
        ConfigLoader instance
    """
    global _config

    if _config is None or reload:
        _config = ConfigLoader()

    return _config


if __name__ == "__main__":
    # Test configuration loading
    logger.add("logs/config.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print("CONFIGURATION LOADER TEST")
    print("=" * 70)

    config = get_config()

    print("\nModel Configuration:")
    print(f"  Embedding model: {config.models.embeddings.model}")
    print(f"  LLM model: {config.models.llm.model}")
    print(f"  Evaluation model: {config.models.evaluation.model}")

    print("\nVRAM Configuration:")
    print(f"  Total: {config.models.vram.total}GB")
    print(f"  Mode: {config.models.vram.mode}")
    print(f"  Headroom: {config.models.vram.headroom}GB")

    print("\nRetrieval Configuration:")
    print(f"  Hybrid search: {config.system.retrieval.hybrid['enabled']}")
    print(f"  Reranking: {config.system.retrieval.reranking['enabled']}")
    print(f"  Max results: {config.system.retrieval.max_results}")

    print("\nChunking Configuration:")
    print(f"  Type: {config.system.chunking.type}")
    print(f"  Chunk size: {config.system.chunking.chunk_size}")
    print(f"  Preserve equations: {config.system.chunking.preserve_equations}")

    print("\nDot-notation access:")
    print(f"  models.embeddings.model = {config.get('models.embeddings.model')}")
    print(f"  system.chunking.chunk_size = {config.get('system.chunking.chunk_size')}")

    print("\n" + "=" * 70)
    print("Configuration loaded successfully! ✅")
    print("=" * 70 + "\n")
