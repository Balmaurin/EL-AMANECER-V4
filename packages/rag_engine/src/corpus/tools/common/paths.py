"""
Centralized absolute path definitions for cross-PC portability.

All paths are computed from PROJECT_ROOT using __file__, ensuring
that the system works from any PC or working directory without
relative path errors.
"""

from pathlib import Path

# Compute PROJECT_ROOT from this file's location
# tools/common/paths.py -> tools/common/ -> tools/ -> project_root/
_CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = _CURRENT_FILE.parent.parent.parent

# Core directories - absolute paths computed from PROJECT_ROOT
INDEX_DIR = PROJECT_ROOT / "index"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TOOLS_DIR = PROJECT_ROOT / "tools"
SERVER_DIR = PROJECT_ROOT / "server"
CORPUS_DIR = PROJECT_ROOT / "corpus"

# Sub-directories for specific functionality
EMBEDDING_DIR = DATA_DIR / "embeddings"
EVAL_DATA_DIR = DATA_DIR / "eval"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
CATALOG_DIR = CORPUS_DIR / "_registry"

# Common file paths
LOG_FILE = LOGS_DIR / "rag.log"
CONFIG_FILE = CONFIG_DIR / "universal.yaml"
CATALOG_DB_PATH = CATALOG_DIR / "catalog.duckdb"
MANIFESTS_PATH = CATALOG_DIR / "manifests.parquet"
CORPUS_MAPPING_PATH = INDEX_DIR / "corpus_mapping.parquet"

# Cache directories
CACHE_DIR = PROJECT_ROOT / ".cache"
CHUNK_CACHE_DIR = CACHE_DIR / "chunks"
EMBED_CACHE_DIR = CACHE_DIR / "embeddings"
SEARCH_CACHE_DIR = CACHE_DIR / "search"

# Index file paths
HNSW_INDEX_PATH = INDEX_DIR / "hnsw.hnswlib"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.ivfpq"
TANTIVY_INDEX_DIR = INDEX_DIR / "tantivy"

# Embedding storage
EMBEDDINGS_PATH = EMBEDDING_DIR / "embeddings.parquet"


def ensure_directories():
    """Create all necessary directories if they don't exist."""
    for dir_path in [
        INDEX_DIR,
        DATA_DIR,
        LOGS_DIR,
        CONFIG_DIR,
        EMBEDDING_DIR,
        EVAL_DATA_DIR,
        CATALOG_DIR,
        CACHE_DIR,
        CHUNK_CACHE_DIR,
        EMBED_CACHE_DIR,
        SEARCH_CACHE_DIR,
        TANTIVY_INDEX_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_corpus_path(corpus_name: str = "universal") -> Path:
    """Get absolute path to a corpus directory."""
    return CORPUS_DIR / corpus_name


def get_index_path(corpus_name: str = "universal") -> Path:
    """Get absolute path to index files for a corpus."""
    return INDEX_DIR / corpus_name


def get_logs_path(log_name: str = "rag.log") -> Path:
    """Get absolute path to a log file."""
    return LOGS_DIR / log_name


if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"INDEX_DIR: {INDEX_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"LOGS_DIR: {LOGS_DIR}")
    print(f"CONFIG_DIR: {CONFIG_DIR}")
    print(f"All paths are absolute: {INDEX_DIR.is_absolute()}")
