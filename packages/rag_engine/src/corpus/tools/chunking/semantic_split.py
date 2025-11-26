"""
Semantic text chunking module for RAG systems.

This module implements semantic-aware text chunking strategies that preserve
context and meaning while splitting documents into manageable pieces.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

# Lazy import hints only for type-checkers
if TYPE_CHECKING:
    import torch
    from sentence_transformers import SentenceTransformer, util
else:
    torch = None
    SentenceTransformer = None
    util = None

# NOTE: NO heavy imports at top-level
from tools.chunking.chunk_cache import ChunkCache
from tools.common.config import load_config  # <- faltaba
from tools.common.errors import ChunkingError
from tools.monitoring.metrics import monitor_operation

# Configure logging
log = logging.getLogger("rag.chunk")


def _safe_filename(s: str) -> str:
    """Make a safe filename from an arbitrary id/title."""
    if not s:
        return "doc"
    # Normalize accents and remove weird chars
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Replace path separators and spaces
    s = s.replace("/", "_").replace("\\", "_").strip()
    # Keep only simple charset
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:200] or "doc"


class SemanticChunker:
    """Advanced semantic chunking with quality optimization and caching"""

    def __init__(
        self, config_path: Optional[str] = None, cache_dir: Optional[str] = None
    ):
        # Load configuration
        if config_path:
            with open(config_path, encoding="utf-8") as f:
                _cfg = yaml.safe_load(f) or {}
        else:
            _cfg = {}

        # Ensure a well-typed config dict for downstream type-checkers
        if not isinstance(_cfg, dict):
            _cfg = {}
        self.config: Dict[str, Any] = _cfg

        # Lazy initialization - embedder loads only when needed
        self._embedder = None
        self._embedder_model: str = str(
            self.config.get("embedder", "sentence-transformers/all-MiniLM-L6-v2")
        )

        # DO NOT import or instantiate ChunkOptimizer here
        self._optimizer = None
        # Optimizer gated (Windows-first, avoid heavy deps by default)
        try:
            import os as _os

            self._opt_enabled = bool(
                self.config.get("optimizer", {}).get("enabled", False)
                or (_os.getenv("RAG_CHUNK_OPTIMIZER_ENABLED", "0") == "1")
            )
        except Exception:
            self._opt_enabled = False

        # Initialize chunk cache
        self.cache = ChunkCache(
            cache_dir=cache_dir or "cache/chunks",
            max_cache_size=self.config.get("max_cache_size", 1_000_000_000),
            vacuum_threshold=self.config.get("vacuum_threshold", 0.8),
        )

    def _get_optimizer(self):
        """Lazy import + instantiate optimizer only if/when needed."""
        if self._optimizer is None:
            # Import here to avoid pulling torch/transformers at module import time
            from tools.chunking.chunk_optimizer import ChunkOptimizer

            self._optimizer = ChunkOptimizer(
                cache_size=self.config.get("cache_size", 10000),
                min_chunk_length=self.config.get("min_chunk_length", 100),
                max_chunk_length=self.config.get("max_chunk_length", 1000),
                target_overlap=self.config.get("target_overlap", 0.2),
            )
        return self._optimizer

    @property
    def optimizer(self):
        """Lightweight proxy exposing optimizer limits without importing heavy deps.

        Tests expect `chunker.optimizer.min_chunk_length` and `.max_chunk_length`.
        """
        try:
            cfg = self.config if isinstance(self.config, dict) else {}
        except Exception:
            cfg = {}
        return SimpleNamespace(
            min_chunk_length=int(cfg.get("min_chunk_length", 100)),
            max_chunk_length=int(cfg.get("max_chunk_length", 1000)),
        )

    @property
    def embedder(self):
        """Lazy load of the embedder - only when used"""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer as ST

                self._embedder = ST(self._embedder_model)
            except OSError as err:
                raise RuntimeError(
                    f"No se pudo cargar el modelo de embeddings {self._embedder_model}. "
                    "Verifica que PyTorch y sus dependencias estén correctamente instalados."
                ) from err
        return self._embedder

    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into optimized semantic chunks with caching.
        """
        with monitor_operation("chunking", "chunk_text") as span:
            try:
                # Generate initial chunks
                initial_chunks = self._generate_initial_chunks(text)
                span.set_attribute("initial_chunks", len(initial_chunks))

                # Try to get chunks from cache
                cached_chunks: List[str] = []
                chunks_to_optimize: List[str] = []
                chunk_indices: List[int] = []

                for i, chunk in enumerate(initial_chunks):
                    chunk_hash = self.cache._generate_hash(chunk)  # type: ignore[reportPrivateUsage]
                    cached_content = self.cache.get_chunk(chunk_hash)

                    if cached_content is not None:
                        cached_chunks.append(cached_content)
                    else:
                        chunks_to_optimize.append(chunk)
                        chunk_indices.append(i)

                span.set_attribute("cache_hits", len(cached_chunks))
                span.set_attribute("chunks_to_optimize", len(chunks_to_optimize))

                # Optimize remaining chunks (lazy import of optimizer), gated by flag
                if self._opt_enabled and chunks_to_optimize:
                    optimized_chunks = self._get_optimizer().optimize_chunks(
                        text, chunks_to_optimize
                    )
                    # Cache optimized chunks
                    for chunk in optimized_chunks:
                        self.cache.store_chunk(
                            chunk=chunk,
                            source_file=(metadata or {}).get("source", ""),
                            quality_score=1.0,  # Optimized chunks get max score
                        )
                else:
                    # If optimizer disabled, we pass-through original chunks
                    optimized_chunks = chunks_to_optimize[:]

                # Merge cached and optimized chunks
                final_chunks: List[str] = []
                opt_idx = 0

                for i in range(len(initial_chunks)):
                    if i in chunk_indices:
                        final_chunks.append(optimized_chunks[opt_idx])
                        opt_idx += 1
                    else:
                        final_chunks.append(cached_chunks[i - opt_idx])

                # Prepare result with metadata
                result: List[Dict[str, Any]] = []
                for chunk in final_chunks:
                    result.append({"text": chunk, "metadata": metadata or {}})

                span.set_attribute("final_chunks", len(result))
                return result

            except Exception as e:
                span.set_status("error")
                span.set_attribute("error", str(e))
                raise ChunkingError(f"Error during chunking: {e}")

    def _generate_initial_chunks(self, text: str) -> List[str]:
        """Generate initial chunks using basic paragraph grouping strategy"""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0
        max_length = int(self.config.get("max_chunk_length", 1000))

        for para in paragraphs:
            para_length = len(para)
            if current_length + para_length > max_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(para)
            current_length += para_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def chunk_file(
        self, file_path: str, encoding: str = "utf-8"
    ) -> List[Dict[str, Any]]:
        """Chunk content of a file"""
        with monitor_operation("chunking", "chunk_file") as span:
            try:
                path = Path(file_path)
                if not path.exists():
                    raise ChunkingError(f"File not found: {path}")

                text = path.read_text(encoding=encoding, errors="replace")

                metadata = {
                    "source": str(path),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "size": path.stat().st_size,
                }

                chunks = self.chunk_text(text, metadata)

                span.set_attribute("file", str(path))
                span.set_attribute("chunks", len(chunks))
                return chunks

            except Exception as e:
                span.set_status("error")
                span.set_attribute("error", str(e))
                raise ChunkingError(f"Error chunking file {file_path}: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the chunk cache"""
        return self.cache.get_stats()


def sentence_split(text: str) -> List[str]:
    """Split text into sentences using punctuation heuristics."""
    if not text or not isinstance(text, str):
        return []
    return re.split(r"(?<=[\.!?])\s+", text.strip())


def semantic_chunks(base: Path):
    """
    Divide el texto en trozos semánticamente coherentes (sin requerir Torch).
    Lee .jsonl (múltiples líneas JSON por archivo) y escribe un .jsonl por chunk.
    """
    # Cargar configuración (si no existe, dict vacío)
    try:
        config = load_config("chunking")
    except Exception:
        config = {}

    semantic_config = config.get("semantic_splitting", {})
    max_length = int(semantic_config.get("max_tokens", 230))
    min_chunk_size = max(60, int(max_length * 0.35))

    out_dir = base / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_docs = 0
    n_chunks = 0

    # Recorre todos los .jsonl del directorio 'cleaned' (salida de normalize)
    cleaned_dir = base / "cleaned"
    for file_path in cleaned_dir.glob("*.jsonl"):
        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except Exception as e:
                        log.warning(f"Línea inválida en {file_path.name}: {e}")
                        continue

                    text = doc.get("text", "")
                    doc_id = doc.get("doc_id") or doc.get("id") or file_path.stem
                    safe_id = _safe_filename(str(doc_id))

                    sentences = sentence_split(text)
                    if not sentences:
                        # sin oraciones, intenta un fallback por párrafos
                        paragraphs = [
                            p.strip() for p in text.split("\n\n") if p.strip()
                        ]
                        sentences = paragraphs if paragraphs else [text] if text else []

                    # Agrupar oraciones en trozos
                    chunks: List[str] = []
                    current_chunk: List[str] = []
                    current_length = 0

                    for sentence in sentences:
                        sentence_length = len(sentence.split())
                        if current_length + sentence_length <= max_length:
                            current_chunk.append(sentence)
                            current_length += sentence_length
                        else:
                            if current_chunk:
                                chunks.append(" ".join(current_chunk))
                            current_chunk = [sentence]
                            current_length = sentence_length

                    if current_chunk:
                        chunks.append(" ".join(current_chunk))

                    # Filtrar trozos demasiado pequeños
                    chunks = [c for c in chunks if len(c.split()) >= min_chunk_size]

                    # Guardar cada trozo como una línea .jsonl
                    for order, chunk in enumerate(chunks):
                        chunk_id = f"{safe_id}#{order}"
                        item = {
                            "doc_id": str(doc_id),
                            "chunk_id": chunk_id,
                            "text": chunk,
                            "order": order,
                            "meta": {
                                "title": doc.get("title", ""),
                                "lang": doc.get("lang", ""),
                                "tags": doc.get("tags", []),
                                "quality": doc.get("quality", 0.5),
                            },
                        }
                        chunk_path = out_dir / f"{safe_id}_{order}.jsonl"
                        with chunk_path.open("w", encoding="utf-8") as out_f:
                            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        n_chunks += 1

                    n_docs += 1

        except Exception as e:
            log.error(f"Error procesando el archivo {file_path}: {e}")
            continue

    log.info(f"Documentos procesados: {n_docs}, Trozos creados: {n_chunks}")
    return {"docs": n_docs, "chunks": n_chunks}


# -----------------------------------------------------------------------------
# Si en el futuro quieres reactivar embeddings para hacer “true semantic merge”,
# descomenta este bloque dentro de semantic_chunks() y añade una flag en config:
#
#     use_st = bool(semantic_config.get("use_sentence_transformers", False))
#     if use_st:
#         import torch as _torch
#         device = "cuda" if _torch.cuda.is_available() else "cpu"
#         try:
#             import torch_directml
#             device = torch_directml.device()  # soporte AMD DirectML si está
#         except Exception:
#             pass
#         from sentence_transformers import SentenceTransformer as _ST
#         st_model = _ST(semantic_config.get("model", "sentence-transformers/all-MiniLM-L6-v2"), device=device)
#         # usar st_model.encode(...) para scoring/agrupado si lo necesitas
# -----------------------------------------------------------------------------
