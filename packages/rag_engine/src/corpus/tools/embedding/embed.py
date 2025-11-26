"""
Text embedding module for RAG systems.

This module handles the conversion of text chunks into dense vector
embeddings using transformer models, with support for:
- Efficient batched processing
- Embedding caching
- GPU acceleration
- Memory optimization
"""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

try:
    # Carga automática de variables desde .env (si existe)
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# Deshabilitar CUDA completamente en Windows para evitar errores de multiprocessing
# Debe estar ANTES de importar torch/transformers
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "0")

# Lazy import de torch y sentence_transformers
if TYPE_CHECKING:
    import torch
    from sentence_transformers import SentenceTransformer
else:
    torch = None
    SentenceTransformer = None

from tqdm import tqdm

from tools.common.device import pick_device
from tools.common.utils import hash_text
from tools.embedding.embed_cache import EmbCache

# Configure logging
log = logging.getLogger("rag.embed")


# Helper: load ST model forcing CPU unless explicitly overridden via env
def _load_st_model(model_name: str):
    """Load SentenceTransformer model forcing CPU by default.

    This avoids device autodetection that may trigger DLL initialization
    issues on Windows. If RAG_DEVICE env var is set to a value other than
    'cpu', the caller can choose a different path explicitly.
    """
    from sentence_transformers import SentenceTransformer  # local import

    device = os.getenv("RAG_DEVICE", "cpu").lower()
    if device != "cpu":
        device = "cpu"
    return SentenceTransformer(model_name, device=device)


def _load_config() -> Dict[str, Any]:
    """Load embedding configuration from universal config file."""
    try:
        with open("config/universal.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        log.error("Configuration file not found")
        raise
    except yaml.YAMLError as e:
        log.error(f"Error parsing configuration: {e}")
        raise


def _get_provider(cfg: Dict[str, Any]) -> str:
    """Get embedding provider from config or env override."""
    env_val = os.getenv("RAG_embedder__provider")
    if env_val:
        return env_val.lower()
    return str(cfg.get("embedder", {}).get("provider", "local")).lower()


def embed_corpus(
    branch: str,
    base: Path,
    show_progress: bool = True,
    batch_size: int = 32,
    retry_attempts: int = 3,
    rebuild: bool = False,
) -> None:
    """Generate embeddings for all chunks in a corpus.

    This function processes text chunks from the given corpus path,
    generates embeddings using the configured model, and stores them
    in the embedding cache.

    Args:
        branch: str
            The branch name for versioning the embeddings
        base: Path
            Base directory containing the text chunks
        show_progress: bool, optional
            Whether to display a progress bar, by default True
        batch_size: int, optional
            Number of chunks to process at once, by default 32
        retry_attempts: int, optional
            Number of retries for failed embeddings, by default 3

    Raises:
        FileNotFoundError: If corpus directory or config file not found
        ValueError: If no text chunks found or invalid configuration
        RuntimeError: If embedding generation fails after all retries

    Args:
        branch: Source branch or version identifier
        base: Base directory containing chunks and where embeddings will be stored
        show_progress: Whether to display progress bars

    Raises:
        FileNotFoundError: If input chunks or config not found
        ValueError: If configuration is invalid
    """
    # Load configuration
    cfg = _load_config()
    embedding_cfg = cfg.get("embedder", {})

    provider = _get_provider(cfg)
    model_name = embedding_cfg.get("model")
    if not model_name:
        raise ValueError("No embedding model specified in config")

    # Determine device label for logging (will be adjusted below)
    device_cfg = embedding_cfg.get("device", "auto")
    device_label = "cpu"

    # Local provider (SentenceTransformers) path
    if provider == "local":
        try:
            # Force CPU path unless the user explicitly overrides via env
            if os.getenv("RAG_DEVICE", "cpu").lower() == "cpu":
                model = _load_st_model(model_name)
                device_label = "cpu"
                log.info(f"Using SentenceTransformer on CPU for {model_name}")
            else:
                # Only if explicitly overridden, honor non-CPU device
                # while still avoiding top-level torch imports
                from sentence_transformers import SentenceTransformer  # local import

                device = pick_device(device_cfg)
                device_label = device
                log.info(f"Using device: {device}")
                model = SentenceTransformer(model_name, device=device)
        except OSError as e:
            log.warning(
                f"PyTorch no disponible para {model_name}: {e}. Se omiten embeddings y se continúa con BM25/otros índices."
            )
            return
        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise

        def encode_fn(texts: List[str]) -> np.ndarray:
            # Usar batch_size pequeño para evitar problemas de memoria
            # y reducir la carga en multiprocessing interno de torch
            return np.asarray(
                model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=8,  # Batch pequeño para Windows
                ),
                dtype=np.float32,
            )

    # Llama.cpp provider (ligero, sin PyTorch, ideal para Windows)
    elif provider == "llamacpp":
        try:
            from llama_cpp import Llama
        except ImportError:
            log.error(
                "llama-cpp-python no instalado. Instalar con: pip install llama-cpp-python"
            )
            raise

        # Buscar modelo GGUF
        model_path = embedding_cfg.get("model_path")
        if not model_path:
            # Ruta por defecto
            model_path = "models/embeddings/nomic-embed-text-v1.5.Q4_0.gguf"

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            log.error(f"Modelo GGUF no encontrado: {model_path}")
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        log.info(f"Cargando modelo llama.cpp: {model_path}")

        # Cargar modelo con embedding=True
        llama_model = Llama(
            model_path=str(model_path_obj),
            embedding=True,
            n_ctx=512,  # Contexto pequeño para embeddings
            n_threads=4,  # 4 threads para Windows
            verbose=False,
        )

        device_label = "llamacpp-cpu"
        log.info(f"Modelo llama.cpp cargado exitosamente")

        def encode_fn(texts: List[str]) -> np.ndarray:
            """Generar embeddings con llama.cpp"""
            vectors = []
            for text in texts:
                # Generar embedding para cada texto
                embedding = llama_model.embed(text)
                vectors.append(np.array(embedding, dtype=np.float32))
            return np.vstack(vectors)

    # OpenAI remote embeddings path
    elif provider == "openai":
        try:
            from openai import OpenAI
        except Exception as e:
            log.error(f"OpenAI SDK not installed: {e}")
            raise
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            log.error("OPENAI_API_KEY not set. Cannot use OpenAI embeddings provider.")
            return
        client = OpenAI(api_key=api_key)
        device_label = "openai"

        def encode_fn(texts: List[str]) -> np.ndarray:
            # OpenAI embeddings API accepts up to 2048 inputs; we'll chunk as needed
            out: List[np.ndarray] = []
            B = 2048
            for i in range(0, len(texts), B):
                batch = texts[i : i + B]
                resp = client.embeddings.create(model=model_name, input=batch)
                out.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
            return np.vstack(out)

        log.info(f"Using OpenAI embeddings provider with model: {model_name}")

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    # Get batch size from env var or config (reducido para Windows)
    batch_size = int(
        os.getenv("RAG_embedder__batch_size", embedding_cfg.get("batch_size", 8))
    )

    # Load chunks
    chunks_dir = base / "chunks"
    if not chunks_dir.exists():
        raise FileNotFoundError("Chunks directory not found")

    rows = []
    for chunk_file in chunks_dir.glob("*.jsonl"):
        try:
            chunk = json.loads(chunk_file.read_text(encoding="utf-8"))
            rows.append(chunk)
        except json.JSONDecodeError as e:
            log.error(f"Error reading chunk file {chunk_file}: {e}")
            continue

    if not rows:
        log.warning("No chunks found for embedding")
        return

    # Create DataFrame and prepare for embedding
    df = pd.DataFrame(rows)
    if "text" not in df.columns:
        raise ValueError("Chunks must contain 'text' field")

    # Initialize cache and vectors
    cache = EmbCache()
    vectors = []
    texts = df["text"].tolist()
    text_hashes = [hash_text(t) for t in texts]

    # Find cache misses
    cache_miss_indices = []
    cache_miss_texts = []

    if rebuild:
        # Force re-embedding of all texts, ignoring cache
        cache_miss_indices = list(range(len(texts)))
        cache_miss_texts = texts[:]
        vectors = [None] * len(texts)
    else:
        for idx, text_hash in enumerate(text_hashes):
            cached_vector = cache.get(text_hash)
            if cached_vector is None:
                cache_miss_indices.append(idx)
                cache_miss_texts.append(texts[idx])
            vectors.append(cached_vector)

    # Process cache misses in batches
    if cache_miss_texts:
        log.info(f"Generating embeddings for {len(cache_miss_texts)} new texts")

        with tqdm(
            total=len(cache_miss_texts),
            desc="Generating embeddings",
            disable=not show_progress,
        ) as pbar:
            for i in range(0, len(cache_miss_texts), batch_size):
                # Get batch of texts
                batch_texts = cache_miss_texts[i : i + batch_size]
                try:
                    # Generate embeddings via selected provider
                    batch_vectors = encode_fn(batch_texts).astype(np.float32)

                    # Update vectors and cache
                    for j, vector in enumerate(batch_vectors):
                        idx = cache_miss_indices[i + j]
                        vectors[idx] = vector
                        cache.store(text_hashes[idx], vector)

                    pbar.update(len(batch_texts))

                except Exception as e:
                    log.error(f"Error generating embeddings for batch: {e}")
                    # Skip failed batch but continue processing
                    continue

    # Close cache
    cache.close()

    # Save embeddings
    df["vector"] = vectors
    output_path = base / "embeddings" / "embeddings.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(output_path, index=False)
        log.info(f"Embeddings saved ({device_label}) -> {output_path}")
    except Exception as e:
        log.error(f"Error saving embeddings: {e}")
        raise


def _load_cfg():
    with open("config/universal.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
