"""
Configuration management module for the RAG system.

This module provides a robust configuration system with:
- Schema validation using Pydantic
- Environment variable overrides
- Hot reload capability
- Version control
- Comprehensive documentation
"""

import copy
import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, confloat, conint, field_validator
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

log = logging.getLogger("rag.config")


def deep_merge(a: dict, b: dict) -> dict:
    """Deep merge two dictionaries.

    Args:
        a: Base dictionary
        b: Dictionary to merge on top

    Returns:
        Merged dictionary
    """
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# Configuration Models
class EmbedderConfig(BaseModel):
    """Embedding model configuration."""

    model: str = Field(
        default="BAAI/bge-m3", description="Model identifier from Hugging Face hub"
    )
    device: str = Field(
        default="auto", description="Device to run model on (auto|cpu|cuda)"
    )
    batch_size: int = Field(
        default=64, ge=1, description="Batch size for embedding generation"
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        if v not in ["auto", "cpu", "cuda"]:
            raise ValueError("Device must be auto, cpu or cuda")
        return v


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""

    mode: str = Field(default="semantic", description="Chunking mode (semantic|simple)")
    target_words: int = Field(
        default=220, ge=1, description="Target chunk size in words"
    )
    overlap_words: int = Field(
        default=40, ge=0, description="Number of words to overlap between chunks"
    )
    min_words: int = Field(default=60, ge=1, description="Minimum chunk size in words")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v not in ["semantic", "simple"]:
            raise ValueError("Mode must be semantic or simple")
        return v

    @field_validator("overlap_words")
    @classmethod
    def validate_overlap(cls, v, info):
        if v >= info.data.get("target_words", 220):
            raise ValueError("Overlap must be less than target size")
        return v


class IndexConfig(BaseModel):
    """Vector and lexical index configuration."""

    backend: str = Field(
        default="hnsw", description="Vector index backend (hnsw|faiss)"
    )
    faiss_factory: str = Field(
        default="IVF4096,PQ32", description="FAISS index factory string"
    )
    faiss_nprobe: int = Field(default=32, ge=1, description="FAISS nprobe parameter")
    bm25_backend: str = Field(
        default="whoosh", description="BM25 backend (whoosh|tantivy)"
    )
    stored_fields: List[str] = Field(
        default=["chunk_id", "doc_id", "title", "lang", "tags"],
        description="Fields to store in BM25 index",
    )

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v):
        if v not in ["hnsw", "faiss"]:
            raise ValueError("Backend must be hnsw or faiss")
        return v

    @field_validator("bm25_backend")
    @classmethod
    def validate_bm25(cls, v):
        if v not in ["whoosh", "tantivy"]:
            raise ValueError("BM25 backend must be whoosh or tantivy")
        return v


class RetrievalConfig(BaseModel):
    """Retrieval and search configuration."""

    top_k: int = Field(default=10, ge=1, description="Number of results to return")
    rerank_enabled: bool = Field(
        default=True, description="Whether to enable reranking"
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Reranking model name",
    )
    fusion_method: str = Field(
        default="rrf", description="Result fusion method (rrf|weighted)"
    )
    fusion_weights: Dict[str, float] = Field(
        default={
            "faiss": 0.45,
            "bm25": 0.25,
            "graph": 0.10,
            "raptor": 0.10,
            "quality": 0.10,
        },
        description="Weights for each retrieval component",
    )

    @field_validator("fusion_method")
    @classmethod
    def validate_fusion(cls, v):
        if v not in ["rrf", "weighted"]:
            raise ValueError("Fusion method must be rrf or weighted")
        return v

    @field_validator("fusion_weights")
    @classmethod
    def validate_weights(cls, v):
        if not abs(sum(v.values()) - 1.0) < 0.001:
            raise ValueError("Fusion weights must sum to 1")
        return v


class RagConfig(BaseModel):
    """Root configuration class."""

    version: str = Field(default="4.0.0", description="Configuration schema version")
    branch: str = Field(default="universal", description="Project branch name")
    embedder: EmbedderConfig = Field(
        default_factory=EmbedderConfig, description="Embedding settings"
    )
    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig, description="Chunking settings"
    )
    index: IndexConfig = Field(
        default_factory=IndexConfig, description="Index settings"
    )
    retrieval: RetrievalConfig = Field(
        default_factory=RetrievalConfig, description="Retrieval settings"
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, v):
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError("Version must be in format X.Y.Z")
        return v


# Configuration Management
_config_instance: Optional[RagConfig] = None
_config_lock = threading.Lock()
_config_watcher = None


class ConfigWatcher:
    """Watch configuration file for changes."""

    def __init__(self, config_path: Union[str, Path], callback):
        self.config_path = Path(config_path)
        self.callback = callback
        self.observer = None

    def start(self):
        """Start watching config file."""

        class ConfigHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.src_path == str(self.config_path.resolve()):
                    try:
                        log.info("Configuration file changed, reloading...")
                        self.callback()
                    except Exception as e:
                        log.error(f"Error reloading config: {e}")

        self.observer = Observer()
        self.observer.schedule(
            ConfigHandler(), str(self.config_path.parent), recursive=False
        )
        self.observer.start()

    def stop(self):
        """Stop watching config file."""
        if self.observer:
            self.observer.stop()
            self.observer.join()


def load_config(path: str = "config/universal.yaml") -> RagConfig:
    """Load and validate configuration.

    Args:
        path: Path to YAML configuration file

    Returns:
        Validated configuration object

    Raises:
        FileNotFoundError: If config file not found
        ValidationError: If config is invalid
    """
    global _config_instance, _config_watcher

    with _config_lock:
        if _config_instance is None:
            # Load base config
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            # Apply environment overrides
            prefix = cfg.get("overrides", {}).get("env_prefix", "RAG_")
            for key, val in os.environ.items():
                if not key.startswith(prefix):
                    continue
                path_keys = key[len(prefix) :].lower().split("__")
                cur = cfg
                for p in path_keys[:-1]:
                    if p not in cur or not isinstance(cur[p], dict):
                        cur[p] = {}
                    cur = cur[p]

                leaf = path_keys[-1]
                # Parse value
                if val.lower() in ("true", "false"):
                    cur[leaf] = val.lower() == "true"
                else:
                    try:
                        # Try to parse as number
                        if "." in val:
                            cur[leaf] = float(val)
                        else:
                            cur[leaf] = int(val)
                    except ValueError:
                        # Fall back to string
                        cur[leaf] = val

            # Validate config
            _config_instance = RagConfig.model_validate(cfg)

            # Start watcher
            if _config_watcher is None:
                _config_watcher = ConfigWatcher(path, _reload_config)
                _config_watcher.start()

    return _config_instance


@lru_cache()
def get_config() -> RagConfig:
    """Get current configuration instance.

    Returns:
        Current validated configuration
    """
    if _config_instance is None:
        load_config()
    return _config_instance


def _reload_config():
    """Reload configuration from disk."""
    global _config_instance
    with _config_lock:
        _config_instance = None
        get_config.cache_clear()
        load_config()
