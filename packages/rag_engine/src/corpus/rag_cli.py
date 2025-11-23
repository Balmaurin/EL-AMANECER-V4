import json
import os
import sys
import time
import traceback
from pathlib import Path

import typer
from typing_extensions import Annotated

"""RAG CLI (Windows-first): robust to Redis/Torch issues.

Small stability tweaks:
- Default to disabling Redis cache unless explicitly enabled (RAG_DISABLE_REDIS)
- Best-effort pre-import of torch to avoid Windows DLL init issues (WinError 1114)
"""

# Default: disable Redis unless user explicitly enables it
os.environ.setdefault("RAG_DISABLE_REDIS", os.getenv("RAG_DISABLE_REDIS", "1"))

# Pre-import torch early to stabilize DLL loading on Windows
try:  # noqa: SIM105
    import torch  # type: ignore  # noqa: F401
except Exception:
    pass

from typing import Any, Callable, Dict, List, Tuple, cast

import yaml

from tools.catalog.catalog import update_catalog
from tools.catalog.catalog import update_manifest as _update_manifest
from tools.chunking.semantic_split import semantic_chunks
from tools.cleaning.normalize import normalize_corpus
from tools.embedding.embed import embed_corpus
from tools.graph.build_graph import build_graph
from tools.index.index_bm25_tantivy import build_tantivy
from tools.index.index_bm25_whoosh import build_bm25
from tools.index.index_hnsw import build_hnsw
from tools.index.index_milvus import upsert_milvus
from tools.index.index_qdrant import upsert_qdrant
from tools.ingest.ingest_folder import ingest_folder
from tools.raptor.build_tree import build_raptor
from tools.retrieval.gating_crag import apply_crag as _apply_crag
from tools.retrieval.search_unified import unified_search

# Static typing shims for external functions lacking precise annotations
ApplyCragFn = Callable[
    [str, Callable[[str, int, str], List[Dict[str, Any]]], List[Any], Dict[str, Any]],
    Tuple[List[Dict[str, Any]], Dict[str, Any]],
]
apply_crag: ApplyCragFn = cast(ApplyCragFn, _apply_crag)

UpdateManifestFn = Callable[[str, Path, Dict[str, Any]], None]
update_manifest: UpdateManifestFn = cast(UpdateManifestFn, _update_manifest)

# Early runtime guards for Windows stability and local runs
try:
    # Prefer disabling Redis by default for CLI searches unless user overrides
    os.environ.setdefault("RAG_DISABLE_REDIS", "1")
    # Pre-import torch on Windows if using local embeddings to avoid DLL init issues
    _cfg_boot = yaml.safe_load(
        Path("config/universal.yaml").read_text(encoding="utf-8")
    )
    if str(_cfg_boot.get("embedder", {}).get("provider", "local")).lower() == "local":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        try:
            import torch  # noqa: F401
        except Exception:
            pass
except Exception:
    pass

app = typer.Typer(help="RAG Universal++ v3 (Windows-first)")
BRANCH = "universal"


def _update_pipeline_status(
    status_file: Path, step: str, status: str = "running", error: str = ""
):
    """Update pipeline status for dashboard tracking"""
    try:
        status_data = {
            "step": step,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": error,
        }
        status_file.write_text(
            json.dumps(status_data, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass  # Don't fail pipeline if status update fails


def _snapshot_path(snapshot: str):
    base = Path("corpus") / BRANCH
    if not snapshot:
        snapshot = time.strftime("%Y-%m-%d_%H%M")
    p = base / snapshot
    for sub in ["raw", "cleaned", "chunks", "embeddings", "index", "datasets", "eval"]:
        (p / sub).mkdir(parents=True, exist_ok=True)
    # pointer file fallback for Windows instead of symlink
    (base / "latest.ptr").write_text(p.name, encoding="utf-8")
    return p


def _latest():
    base = Path("corpus") / BRANCH
    ptr = base / "latest.ptr"
    if ptr.exists():
        target = base / ptr.read_text(encoding="utf-8").strip()
        if target.exists():
            return target
    snaps = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name != "latest"], reverse=True
    )
    return snaps[0]


@app.command()
def ingest(
    src: Annotated[str, typer.Argument(help="Source folder path")],
    snapshot: Annotated[str, typer.Argument(help="Snapshot name")] = "",
):
    """Ingest documents from a folder"""
    base = _snapshot_path(snapshot)
    ingest_folder(Path(src), base / "raw")
    typer.echo(f"[OK] Ingesta -> {base/'raw'}")


@app.command()
def pipeline(
    snapshot: Annotated[str, typer.Argument(help="Snapshot name")] = "",
    incremental: Annotated[
        bool,
        typer.Option(
            "--incremental/--full",
            help="Incremental (only new/modified) or full rebuild",
        ),
    ] = True,
    force: Annotated[
        bool, typer.Option("--force", help="Force full rebuild (ignore manifest)")
    ] = False,
):
    """Run RAG pipeline (incremental by default, processes only new/modified files)"""
    base = _snapshot_path(snapshot)
    status_file = base / ".pipeline_status.json"

    try:
        # Import incremental normalizer
        from tools.cleaning.normalize_incremental import normalize_corpus_incremental

        # Normalize and chunk
        _update_pipeline_status(status_file, "normalization", "running")
        if incremental and not force:
            typer.echo(
                "[*] Pipeline incremental activado (solo archivos nuevos/modificados)"
            )
            stats_norm: Dict[str, Any] = normalize_corpus_incremental(base)

            # Mostrar estadísticas incrementales
            typer.echo(f"  [+] Nuevos: {stats_norm.get('docs_new', 0)}")
            typer.echo(f"  [~] Actualizados: {stats_norm.get('docs_updated', 0)}")
            typer.echo(f"  [-] Eliminados: {stats_norm.get('docs_removed', 0)}")
            typer.echo(f"  [=] Sin cambios: {stats_norm.get('docs_skipped', 0)}")

            # Solo ejecutar chunking si hay cambios
            total_changes = (
                stats_norm.get("docs_new", 0)
                + stats_norm.get("docs_updated", 0)
                + stats_norm.get("docs_removed", 0)
            )
            _update_pipeline_status(status_file, "normalization", "completed")

            _update_pipeline_status(status_file, "chunking", "running")
            if total_changes > 0:
                stats_chunks: Dict[str, Any] = semantic_chunks(base)
            else:
                typer.echo("  [=] Sin cambios detectados, omitiendo chunking")
                stats_chunks = {"chunks_created": 0}
            _update_pipeline_status(status_file, "chunking", "completed")
        else:
            typer.echo("[*] Pipeline completo activado (rebuild total)")
            stats_norm = normalize_corpus(base)
            _update_pipeline_status(status_file, "normalization", "completed")

            _update_pipeline_status(status_file, "chunking", "running")
            stats_chunks = semantic_chunks(base)
            _update_pipeline_status(status_file, "chunking", "completed")

        update_catalog(BRANCH, base)

        # Embeddings (usa cache automáticamente si rebuild=False)
        _update_pipeline_status(status_file, "embeddings", "running")
        embed_corpus(BRANCH, base, rebuild=force)
        _update_pipeline_status(status_file, "embeddings", "completed")

        # Choose vector index backend
        _update_pipeline_status(status_file, "indexing", "running")
        cfg = yaml.safe_load(Path("config/universal.yaml").read_text(encoding="utf-8"))
        backend = cfg.get("index", {}).get("backend", "hnsw").lower()
        if backend == "faiss":
            from tools.index.index_faiss import build_faiss

            build_faiss(BRANCH, base)
        else:
            build_hnsw(base)

        # Lexical index (Whoosh by default)
        build_bm25(base)
        _maybe_upsert_external(base)

        # Tantivy optional if configured as lexical backend
        try:
            cfg2 = yaml.safe_load(open("config/universal.yaml", "r", encoding="utf-8"))
            if (
                cfg2.get("retrieval", {})
                .get("lexical", {})
                .get("backend", "whoosh")
                .lower()
                == "tantivy"
            ):
                build_tantivy(base)
        except Exception as _e:
            print(f"[warn] Tantivy no disponible: {_e}")
        _update_pipeline_status(status_file, "indexing", "completed")

        # Optional higher-level structures
        _update_pipeline_status(status_file, "raptor", "running")
        build_raptor(base)
        _update_pipeline_status(status_file, "raptor", "completed")

        _update_pipeline_status(status_file, "graph", "running")
        build_graph(base)
        _update_pipeline_status(status_file, "graph", "completed")

        # Update manifest with stats
        update_manifest(BRANCH, base, {**(stats_norm or {}), **(stats_chunks or {})})

        _update_pipeline_status(status_file, "completed", "success")
        if incremental and not force:
            typer.echo(f"[OK] Pipeline Incremental OK -> {base}")
        else:
            typer.echo(f"[OK] Pipeline Universal++ OK -> {base}")
    except Exception as e:
        _update_pipeline_status(status_file, "error", "failed", str(e))
        traceback.print_exc()
        typer.secho(f"[ERROR] Error en pipeline: {e}", err=True, fg=typer.colors.RED)
        sys.exit(1)


@app.command()
def embed(
    snapshot: Annotated[str, typer.Argument(help="Snapshot name")] = "",
    rebuild: Annotated[
        bool, typer.Option(help="Re-embed all texts ignoring cache")
    ] = False,
    device: Annotated[
        str, typer.Option("--device", help="Embedding device override: cpu|cuda|auto")
    ] = "",
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Show extra diagnostics")
    ] = False,
):
    """Generate embeddings only for a snapshot (no indexing)."""
    base = _snapshot_path(snapshot) if snapshot else _latest()
    prev_dev = os.getenv("RAG_DEVICE")
    try:
        # Per-run override for embedding device
        if device:
            os.environ["RAG_DEVICE"] = device
        if verbose:
            # Show provider/model details
            cfg = yaml.safe_load(
                Path("config/universal.yaml").read_text(encoding="utf-8")
            )
            emb = cfg.get("embedder", {})
            eff_dev = os.getenv("RAG_DEVICE", emb.get("device", "auto"))
            typer.echo(
                f"[embed] provider={emb.get('provider','local')} model={emb.get('model','?')} device={eff_dev} rebuild={rebuild}"
            )
        embed_corpus(BRANCH, base, rebuild=rebuild)
        typer.echo(f"[OK] Embeddings -> {base/'embeddings'/'embeddings.parquet'}")
    except Exception as e:
        traceback.print_exc()
        typer.secho(f"[ERROR] Error en embeddings: {e}", err=True, fg=typer.colors.RED)
        sys.exit(1)
    finally:
        # Restore previous device override
        if device:
            if prev_dev is None:
                os.environ.pop("RAG_DEVICE", None)
            else:
                os.environ["RAG_DEVICE"] = prev_dev


@app.command()
def search(
    q: Annotated[str, typer.Argument(help="Search query")],
    top_k: Annotated[int, typer.Argument(help="Number of results")] = 10,
    snapshot: Annotated[str, typer.Argument(help="Snapshot name")] = "",
    mode: Annotated[str, typer.Argument(help="Search mode")] = "hybrid",
    rerank: Annotated[
        bool, typer.Option(help="Enable/disable reranking override for this run")
    ] = True,
    rerank_top_n: Annotated[
        int,
        typer.Option(
            "--rerank-top-n",
            help="How many candidates to send to the reranker (override)",
        ),
    ] = 0,
    rerank_device: Annotated[
        str,
        typer.Option("--rerank-device", help="Reranker device override: cpu|cuda|auto"),
    ] = "",
    crag: Annotated[bool, typer.Option(help="Enable/disable CRAG")] = True,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Show extra diagnostics")
    ] = False,
):
    """Search the RAG corpus"""
    base = _latest() if not snapshot else (Path("corpus") / BRANCH / snapshot)
    try:
        # Per-run override for reranker (A/B testing without editing YAML)
        prev_rr_env = os.getenv("RAG_retrieval__rerank__enabled")
        os.environ["RAG_retrieval__rerank__enabled"] = "true" if rerank else "false"
        prev_rr_topn_env = os.getenv("RAG_retrieval__rerank__top_n")
        prev_rr_dev_env = os.getenv("RAG_retrieval__rerank__device_hint")
        if int(rerank_top_n or 0) > 0:
            os.environ["RAG_retrieval__rerank__top_n"] = str(int(rerank_top_n))
        if rerank_device:
            os.environ["RAG_retrieval__rerank__device_hint"] = rerank_device
        # Windows stability: pre-import torch early to avoid DLL init issues when using local dense search
        if mode.lower() != "bm25":
            try:
                cfg_probe = yaml.safe_load(
                    Path("config/universal.yaml").read_text(encoding="utf-8")
                )
                if (
                    str(cfg_probe.get("embedder", {}).get("provider", "local")).lower()
                    == "local"
                ):
                    import torch

                    _torch = (
                        torch  # mark as used for type checkers; also triggers DLL load
                    )
            except Exception:
                pass

        def retrieve_fn(
            query: str, top_k: int = top_k, mode: str = mode
        ) -> List[Dict[str, Any]]:
            return unified_search(BRANCH, base, query, top_k=top_k, mode=mode)

        if crag:
            try:
                from yaml import safe_load

                cfg = safe_load(
                    Path("config/universal.yaml").read_text(encoding="utf-8")
                )
                crag_cfg = cfg.get("retrieval", {}).get("crag", {})
                fb = crag_cfg.get("fallbacks", [])
                th = crag_cfg.get("thresholds", {})
                if fb and th:
                    hits, info = apply_crag(q, retrieve_fn, fb, th)
                    if verbose:
                        typer.echo(
                            f"CRAG path: {info['path']}  conf: {info['conf']:.3f}"
                        )
                else:
                    hits = retrieve_fn(q)
            except Exception:
                hits = retrieve_fn(q)
        else:
            hits = retrieve_fn(q)

        if verbose:
            # Print configuration and diagnostics
            try:
                cfg = yaml.safe_load(
                    Path("config/universal.yaml").read_text(encoding="utf-8")
                )
                emb = cfg.get("embedder", {})
                retr = cfg.get("retrieval", {})
                rer = retr.get("rerank", {})
                lex = retr.get("lexical", {})
                ext = cfg.get("external_index", {}).get("type", "local")
                idx_backend = cfg.get("index", {}).get("backend", "hnsw")
                redis_disabled = os.getenv("RAG_DISABLE_REDIS", "1") == "1"
                # Detect local vector index files present
                index_dir = base / "index"
                candidates = [
                    "hnsw.idx",
                    "faiss.index",
                    "index.faiss",
                    "corpus_faiss.index",
                ]
                present = [c for c in candidates if (index_dir / c).exists()]
                reranked_by = (
                    hits[0].get("meta", {}).get("reranked_by", "unknown")
                    if hits
                    else "unknown"
                )
                rr_topn = int(
                    os.getenv(
                        "RAG_retrieval__rerank__top_n",
                        str(retr.get("rerank", {}).get("top_n", 0)),
                    )
                    or 0
                )
                rr_dev = os.getenv(
                    "RAG_retrieval__rerank__device_hint", rer.get("device_hint", "cpu")
                )
                typer.echo(
                    "\n".join(
                        [
                            f"[cfg] embedder={emb.get('provider','local')} model={emb.get('model','?')} device={emb.get('device','auto')}",
                            f"[cfg] index.backend={idx_backend} external_index.type={ext} lexical.backend={lex.get('backend','whoosh')}",
                            f"[cfg] retrieval.rerank.enabled={rer.get('enabled',True)} model={rer.get('model','BAAI/bge-reranker-base')} device={rr_dev} top_n={rr_topn}",
                            f"[cfg] retrieval.crag.enabled={retr.get('crag',{}).get('enabled',False)} thresholds={retr.get('crag',{}).get('thresholds',{})}",
                            f"[sys] Redis disabled={redis_disabled} index_files_present={present}",
                            f"[run] mode={mode} reranked_by={reranked_by}",
                        ]
                    )
                )
            except Exception:
                pass

        for i, h in enumerate(hits, 1):
            try:
                src = h.get("source") or (
                    "bm25"
                    if mode == "bm25"
                    else "dense" if mode == "dense" else "hybrid"
                )
                # Limpiar texto de caracteres Unicode problemáticos
                text = (
                    str(h.get("text", ""))
                    .encode("ascii", errors="ignore")
                    .decode("ascii")
                )
                title = (
                    str(h.get("title", ""))
                    .encode("ascii", errors="ignore")
                    .decode("ascii")
                )
                print(
                    f"{i}. score={float(h.get('score', 0.0)):.4f} src={src} "
                    f"[{h.get('chunk_id','')}] {title} -> {text[:160]}..."
                )
            except Exception:
                print(f"{i}. {h}")
        # restore env override
        if prev_rr_env is None:
            os.environ.pop("RAG_retrieval__rerank__enabled", None)
        else:
            os.environ["RAG_retrieval__rerank__enabled"] = prev_rr_env
        if prev_rr_topn_env is None:
            os.environ.pop("RAG_retrieval__rerank__top_n", None)
        else:
            os.environ["RAG_retrieval__rerank__top_n"] = prev_rr_topn_env
        if prev_rr_dev_env is None:
            os.environ.pop("RAG_retrieval__rerank__device_hint", None)
        else:
            os.environ["RAG_retrieval__rerank__device_hint"] = prev_rr_dev_env
    except Exception as e:
        traceback.print_exc()
        typer.secho(f"[ERROR] Error en busqueda: {e}", err=True, fg=typer.colors.RED)
        sys.exit(1)


@app.command()
def info():
    base = Path("corpus") / BRANCH
    snaps = sorted(
        [d.name for d in base.iterdir() if d.is_dir() and d.name != "latest"],
        reverse=True,
    )
    print("Snapshots:", snaps[:10])
    ptr = base / "latest.ptr"
    if ptr.exists():
        print("latest ->", (base / ptr.read_text(encoding="utf-8").strip()).resolve())


@app.command()
def doctor():
    """Environment diagnostics: versions, config highlights, CUDA/Redis status."""
    try:
        import importlib

        py = sys.version.replace("\n", " ")
        try:
            import torch

            torch_v = torch.__version__
            cuda = getattr(torch, "cuda", None)
            cuda_ok = bool(cuda and cuda.is_available())
        except Exception:
            torch_v = "not-imported"
            cuda_ok = False

        def ver(mod: str) -> str:
            try:
                m = importlib.import_module(mod)
                return getattr(m, "__version__", "?")
            except Exception:
                return "not-installed"

        cfg = yaml.safe_load(Path("config/universal.yaml").read_text(encoding="utf-8"))
        print("Python:", py)
        print("Torch:", torch_v, "CUDA:", cuda_ok)
        print("sentence-transformers:", ver("sentence_transformers"))
        print("transformers:", ver("transformers"))
        print("numpy:", ver("numpy"), "pandas:", ver("pandas"))
        print(
            "whoosh:", ver("whoosh"), "hnswlib:", ver("hnswlib"), "faiss:", ver("faiss")
        )
        emb = cfg.get("embedder", {})
        print("embedder:", emb)
        print("retrieval.rerank:", cfg.get("retrieval", {}).get("rerank", {}))
        print("retrieval.crag:", cfg.get("retrieval", {}).get("crag", {}))
        print("retrieval.lexical:", cfg.get("retrieval", {}).get("lexical", {}))
        print("index.backend:", cfg.get("index", {}).get("backend", "hnsw"))
        print(
            "external_index.type:", cfg.get("external_index", {}).get("type", "local")
        )
        print("RAG_DISABLE_REDIS:", os.getenv("RAG_DISABLE_REDIS", "1"))
        print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
    except Exception as e:
        traceback.print_exc()
        typer.secho(f"[ERROR] doctor failed: {e}", err=True, fg=typer.colors.RED)


# --- Quick toggles ---------------------------------------------------------


@app.command()
def enable(
    rerank: Annotated[bool, typer.Option(help="Enable reranker in config")] = True,
    crag: Annotated[bool, typer.Option(help="Enable CRAG in config")] = True,
):
    """Enable RAG features (rerank, CRAG) in config/universal.yaml.

    Este comando ajusta los flags en la configuración y muestra un resumen del estado final.
    """
    try:
        cfg_path = Path("config/universal.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        changed = False
        # Ensure retrieval block
        retrieval = cfg.get("retrieval", {})
        rr = retrieval.get("rerank", {})
        cg = retrieval.get("crag", {})
        if rr.get("enabled") is not True and rerank:
            rr["enabled"] = True
            changed = True
        if cg.get("enabled") is not True and crag:
            cg["enabled"] = True
            changed = True
        # Write back if changed
        retrieval["rerank"] = rr
        retrieval["crag"] = cg
        cfg["retrieval"] = retrieval
        if changed:
            cfg_path.write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            typer.echo("[OK] RAG habilitado en config (rerank/crag)")
        else:
            typer.echo("RAG ya estaba habilitado (rerank/crag)")
        # Print current values
        typer.echo(
            f"rerank.enabled={cfg['retrieval'].get('rerank',{}).get('enabled')}  crag.enabled={cfg['retrieval'].get('crag',{}).get('enabled')}"
        )
    except Exception as e:
        traceback.print_exc()
        typer.secho(f"[ERROR] enable failed: {e}", err=True, fg=typer.colors.RED)


# --- Enterprise-aware pipeline (ingesta paralela + calidad/dedup) ---


@app.command()
def disable(
    rerank: Annotated[bool, typer.Option(help="Disable reranker in config")] = True,
    crag: Annotated[bool, typer.Option(help="Disable CRAG in config")] = True,
):
    """Disable RAG features (rerank, CRAG) in config/universal.yaml.

    Ajusta flags en la configuración y muestra un resumen del estado final.
    """
    try:
        cfg_path = Path("config/universal.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        changed = False
        retrieval = cfg.get("retrieval", {})
        rr = retrieval.get("rerank", {})
        cg = retrieval.get("crag", {})
        if rr.get("enabled") is not False and rerank:
            rr["enabled"] = False
            changed = True
        if cg.get("enabled") is not False and crag:
            cg["enabled"] = False
            changed = True
        retrieval["rerank"] = rr
        retrieval["crag"] = cg
        cfg["retrieval"] = retrieval
        if changed:
            cfg_path.write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            typer.echo("[OK] RAG deshabilitado en config (rerank/crag)")
        else:
            typer.echo("RAG ya estaba deshabilitado (rerank/crag)")
        typer.echo(
            f"rerank.enabled={cfg['retrieval'].get('rerank',{}).get('enabled')}  crag.enabled={cfg['retrieval'].get('crag',{}).get('enabled')}"
        )
    except Exception as e:
        traceback.print_exc()
        typer.secho(f"[ERROR] disable failed: {e}", err=True, fg=typer.colors.RED)


@app.command()
def pipeline_ent(snapshot: Annotated[str, typer.Argument(help="Snapshot name")] = ""):
    """Run enterprise pipeline with parallel ingestion"""
    base = _snapshot_path(snapshot)
    cfg = yaml.safe_load(Path("config/universal.yaml").read_text(encoding="utf-8"))
    ent = cfg.get("enterprise", {})
    w = int(ent.get("ingest", {}).get("workers", 4))
    max_mb = int(ent.get("ingest", {}).get("max_file_mb", 128))
    stats_ing = ingest_folder(
        Path("sample_data"),
        base / "raw",
        workers=w,
        max_file_mb=max_mb,
        skip_hidden=bool(ent.get("ingest", {}).get("skip_hidden", True)),
    )
    print(f"[ingest] {stats_ing}")
    stats_norm: Dict[str, Any] = normalize_corpus(base)
    stats_chunks: Dict[str, Any] = semantic_chunks(base)
    print(f"[normalize] {stats_norm}")
    print(f"[chunks] {stats_chunks}")
    update_catalog(BRANCH, base)
    embed_corpus(BRANCH, base)
    # vector index
    backend = cfg.get("index", {}).get("backend", "hnsw")
    if backend == "faiss":
        from tools.index.index_faiss import build_faiss

        build_faiss(BRANCH, base)
    else:
        build_hnsw(base)
    # lexical
    build_bm25(base)
    _maybe_upsert_external(base)
    try:
        build_tantivy(base)
    except Exception:
        pass
    # point latest
    (base.parent / "latest.ptr").write_text(base.name, encoding="utf-8")
    print("[OK] ENT pipeline listo")


def _maybe_upsert_external(base: Path):
    import yaml

    cfg = yaml.safe_load(Path("config/universal.yaml").read_text(encoding="utf-8"))
    ext = cfg.get("external_index", {}).get("type", "local").lower()
    if ext == "qdrant":
        q = cfg["external_index"]["qdrant"]
        upsert_qdrant(
            base,
            collection=q.get("collection", "rag_universal"),
            host=q.get("host", "127.0.0.1"),
            port=int(q.get("port", 6333)),
        )
    elif ext == "milvus":
        m = cfg["external_index"]["milvus"]
        upsert_milvus(
            base,
            collection=m.get("collection", "rag_universal"),
            host=m.get("host", "127.0.0.1"),
            port=int(m.get("port", 19530)),
            user=m.get("user", "root"),
            password=m.get("password", ""),
            secure=bool(m.get("secure", False)),
        )


if __name__ == "__main__":
    app()
