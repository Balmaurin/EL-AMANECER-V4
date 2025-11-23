from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Enum,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

rag_queries_total = Counter("rag_queries_total", "Total de consultas RAG")
rag_search_seconds = Histogram("rag_search_seconds", "Tiempo de búsqueda RAG (s)")
rag_errors_total = Counter("rag_errors_total", "Errores en /ask")
rag_rerank_used_total = Counter(
    "rag_rerank_used_total", "Veces que se aplicó rerank", ["type"]
)

rag_mode_total = Counter(
    "rag_mode_total", "Consultas por modo de recuperación", ["mode"]
)
rag_context_chars = Histogram(
    "rag_context_chars",
    "Caracteres totales devueltos en el contexto de /ask",
    buckets=[0, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000],
)

rag_lexical_backend = Enum(
    "rag_lexical_backend", "Backend léxico activo", states=["whoosh", "tantivy"]
)
rag_index_backend = Enum(
    "rag_index_backend", "Backend vectorial activo", states=["hnsw", "faiss"]
)
rag_vector_store = Enum(
    "rag_vector_store", "Vector store en uso", states=["local", "qdrant", "milvus"]
)

# Evaluation gauges (optional)
rag_recall_at_10 = Gauge("rag_recall_at_10", "Recall@10 sobre golden set")
rag_mrr = Gauge("rag_mrr", "Mean Reciprocal Rank sobre golden set")
rag_ndcg_10 = Gauge("rag_ndcg_10", "nDCG@10 sobre golden set")


def set_backends(lexical: str, vector: str):
    lexical = lexical if lexical in ("whoosh", "tantivy") else "whoosh"
    vector = vector if vector in ("hnsw", "faiss") else "hnsw"
    rag_lexical_backend.state(lexical)
    rag_index_backend.state(vector)


def metrics_app():
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)
