"""
Dense vector search routing module.

This module handles the routing of dense vector search requests to the
appropriate backend implementation based on configuration. Supported backends:
- Local HNSW index
- Qdrant vector database
- Milvus vector database

The module automatically selects the appropriate backend based on the
configuration in universal.yaml.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Configure logging
log = logging.getLogger("rag.retrieval.dense")


class VectorStore(str, Enum):
    """Supported vector store backends."""

    LOCAL = "local"
    QDRANT = "qdrant"
    MILVUS = "milvus"

    @classmethod
    def from_string(cls, value: str) -> "VectorStore":
        """Create enum from string value."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.LOCAL


def _load_config() -> Dict[str, Any]:
    """Load search configuration from universal config file.

    Returns:
        Dictionary containing search configuration

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If config file is invalid
    """
    try:
        with open("config/universal.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        log.error("Configuration file not found")
        raise
    except yaml.YAMLError as e:
        log.error(f"Error parsing configuration: {e}")
        raise


def dense_search(base: Path, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Execute dense vector search using configured backend.

    Args:
        base: Base directory containing vector indices
        query: Search query string
        top_k: Number of results to return

    Returns:
        List of search results sorted by vector similarity

    Raises:
        ValueError: If query is empty
        ImportError: If required backend module not found
    """
    # Validate parameters
    if not query:
        log.warning("Empty query provided")
        return []

    # Load configuration
    try:
        cfg = _load_config()
        store_type = cfg.get("external_index", {}).get("type", "local")
        store = VectorStore.from_string(store_type)
    except Exception as e:
        log.error(f"Error loading configuration: {e}")
        store = VectorStore.LOCAL

    try:
        # Route to appropriate backend
        if store == VectorStore.QDRANT:
            from .search_qdrant import search_qdrant

            results = search_qdrant(base, query, top_k=top_k)
            log.info(f"Qdrant search returned {len(results)} results")
            return results

        elif store == VectorStore.MILVUS:
            from .search_milvus import search_milvus

            results = search_milvus(base, query, top_k=top_k)
            log.info(f"Milvus search returned {len(results)} results")
            return results

        else:  # VectorStore.LOCAL
            from .search_local_hnsw import search_local_hnsw

            results = search_local_hnsw(base, query, top_k=top_k)
            log.info(f"Local HNSW search returned {len(results)} results")
            return results

    except ImportError as e:
        log.error(f"Required backend module not found: {e}")
        raise
    except Exception as e:
        log.error(f"Error executing dense search: {e}")
        return []
