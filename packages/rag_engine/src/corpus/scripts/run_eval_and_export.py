import os
from pathlib import Path

from server.metrics import rag_mrr, rag_ndcg_10, rag_recall_at_10

from tools.eval.golden_loader import load_golden
from tools.eval.metrics_eval import mrr, ndcg_at_k, recall_at_k
from tools.retrieval.search_unified import unified_search


def main():
    base = Path(os.getenv("RAG_BASE", "corpus/universal"))
    ptr = base / "latest.ptr"
    if ptr.exists():
        snap = ptr.read_text(encoding="utf-8").strip()
        base = base / snap if (base / snap).exists() else base

    golden = Path(os.getenv("RAG_GOLD", "data/eval/dev.jsonl"))
    if not golden.exists():
        print(f"Golden no encontrado: {golden}")
        return

    r1_vals, r5_vals, r10_vals, mrr_vals, ndcg_vals = [], [], [], [], []
    for row in load_golden(golden):
        q = row["question"]
        gold = row["gold"]
        hits = unified_search("universal", base, q, top_k=10, mode="hybrid")
        r1_vals.append(recall_at_k(hits, gold, 1))
        r5_vals.append(recall_at_k(hits, gold, 5))
        r10_vals.append(recall_at_k(hits, gold, 10))
        mrr_vals.append(mrr(hits, gold))
        ndcg_vals.append(ndcg_at_k(hits, gold, 10))

    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    metrics = {
        "recall@10": _avg(r10_vals),
        "mrr": _avg(mrr_vals),
        "ndcg@10": _avg(ndcg_vals),
    }
    rag_recall_at_10.set(metrics["recall@10"])  # type: ignore
    rag_mrr.set(metrics["mrr"])  # type: ignore
    rag_ndcg_10.set(metrics["ndcg@10"])  # type: ignore
    print(metrics)


if __name__ == "__main__":
    main()
