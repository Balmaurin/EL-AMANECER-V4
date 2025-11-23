import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from server.mw.monitoring import MonitoringMiddleware
from server.mw.rate_limit import RateLimitMiddleware
from server.mw.rate_limit_redis import RedisRateLimitMiddleware
from server.mw.request_id import RequestIdMiddleware
from server.security import require_api_key

from tools.common.config import load_config
from tools.common.logging_conf import setup_logging
from tools.common.paths import CORPUS_DIR
from tools.monitoring.alerts import AlertManager
from tools.monitoring.metrics import init_monitoring, monitor_operation
from tools.retrieval.search_unified import unified_search

load_dotenv()

# Load configurations
cfg = load_config()

# Initialize monitoring systems
monitor = init_monitoring(
    prometheus_port=9091,
    service_name="rag-service",  # Default Prometheus port  # Default service name
)

# Initialize alert manager with default configuration
alert_manager = AlertManager({"enabled": False})  # Disabled by default for testing

# Setup logging
logger = setup_logging()


def _latest():
    """Get the latest corpus snapshot using absolute paths."""
    p = CORPUS_DIR / "universal"
    ptr = p / "latest.ptr"
    if ptr.exists():
        target = p / ptr.read_text(encoding="utf-8").strip()
        if (p / target).exists():
            return p / target
        if Path(target).exists():
            return Path(target)
    snaps = sorted(
        [d for d in p.iterdir() if d.is_dir() and d.name != "latest"], reverse=True
    )
    if not snaps:
        raise HTTPException(status_code=404, detail="No snapshots available")
    return snaps[0]


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI app."""
    # Start up monitoring
    monitor.start()
    yield
    # Shut down monitoring
    monitor.stop()


app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)

# Add middlewares
app.add_middleware(MonitoringMiddleware)
app.add_middleware(RateLimitMiddleware, requests=60, per_seconds=60)

# Configure redis-based rate limiter if enabled
redis_url = os.getenv("REDIS_URL")
if redis_url:
    app.add_middleware(
        RedisRateLimitMiddleware, redis_url=redis_url, requests=60, per_seconds=60
    )

app.add_middleware(RequestIdMiddleware)


@app.get("/health")
async def health():
    """Health check endpoint"""
    from datetime import datetime, timezone

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "rag-api",
        "version": "1.0.0",
    }


@app.get("/metrics")
async def metrics():
    """Endpoint para métricas de Prometheus"""
    with monitor_operation("metrics", "collect"):
        from prometheus_client import generate_latest

        return Response(generate_latest(), media_type="text/plain")


@app.get("/ask")
async def ask(
    q: str = Query(..., min_length=2, max_length=512),
    top_k: int = Query(6, ge=1, le=50),
    mode: str = Query("hybrid", pattern="^(hybrid|bm25|graph|expanded)$"),
    rerank: bool = True,
    doc_id: str = Query(None, description="Filter by document ID"),
    doc_title: str = Query(None, description="Filter by document title"),
    min_score: float = Query(
        0.0, ge=0.0, le=1.0, description="Minimum score threshold"
    ),
    page: int = Query(1, ge=1, description="Page number for pagination"),
    page_size: int = Query(10, ge=1, le=100, description="Results per page"),
    expand_query: bool = Query(False, description="Enable query expansion"),
    use_cache: bool = Query(True, description="Use result cache"),
    _=Depends(require_api_key),
):
    """Enhanced search endpoint with filtering, pagination, query expansion, and caching."""
    with monitor_operation("search", "query") as span:
        try:
            span.set_attribute("query", q)
            span.set_attribute("mode", mode)
            span.set_attribute("rerank", rerank)
            span.set_attribute("top_k", top_k)
            span.set_attribute(
                "filters",
                {"doc_id": doc_id, "doc_title": doc_title, "min_score": min_score},
            )
            span.set_attribute("pagination", {"page": page, "page_size": page_size})

            base = _latest()

            # Execute search with query expansion if enabled
            if expand_query:
                from tools.retrieval.query_expansion import get_query_expander

                expander = get_query_expander(use_local_llm=False)
                hits = expander.expand_and_search(
                    lambda q, top_k: unified_search(
                        "universal", base, q, top_k=top_k, mode=mode
                    ),
                    q,
                    top_k=top_k * 2,  # Fetch more to account for filtering
                    max_expansions=3,
                )
                span.set_attribute("query_expansion", True)
            else:
                hits = unified_search("universal", base, q, top_k=top_k * 2, mode=mode)
                span.set_attribute("query_expansion", False)

            # Apply filters
            filtered_hits = []
            for hit in hits:
                # Document ID filter
                if doc_id and hit.get("doc_id", "").lower() != doc_id.lower():
                    continue

                # Document title filter
                if doc_title and doc_title.lower() not in hit.get("title", "").lower():
                    continue

                # Score threshold filter
                if hit.get("score", 0.0) < min_score:
                    continue

                filtered_hits.append(hit)

            # Apply pagination
            total = len(filtered_hits)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_hits = filtered_hits[start_idx:end_idx]

            span.set_attribute("num_results", len(paginated_hits))
            span.set_attribute("total_results", total)
            span.set_attribute("page", page)
            span.set_attribute("page_size", page_size)

            context = "\n\n---\n\n".join(
                [h.get("title", "") + ": " + h.get("text", "") for h in paginated_hits]
            )

            # Reranking temporarily disabled for testing
            # if rerank:
            #     device_hint = "cpu"  # Default device
            #     with monitor_operation("search", "rerank") as rerank_span:
            #         rr = build_reranker(device_hint=device_hint)
            #         hits = rr.rerank(q, paginated_hits, top_k)
            #         rerank_span.set_attribute("reranker_type", rr.__class__.__name__)

            # Record context size
            span.set_attribute("context_chars", len(context))

            return {
                "query": q,
                "mode": mode,
                "filters_applied": {
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "min_score": min_score,
                    "query_expansion": expand_query,
                },
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "pages": (total + page_size - 1) // page_size,
                },
                "results": paginated_hits,
                "prompt_for_llm": f"Usa solo este contexto:\n{context}\n\nPregunta: {q}\nRespuesta:",
            }

        except Exception as e:
            # Alert on search errors
            alert_manager.send_alert(
                "SearchError",
                "error",
                f"Search failed: {str(e)}",
                {"query": q, "mode": mode, "error": str(e)},
            )
            raise HTTPException(status_code=500, detail=f"search_error: {e}")
