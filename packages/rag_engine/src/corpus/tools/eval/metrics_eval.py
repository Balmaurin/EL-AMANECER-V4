import math
from typing import Dict, List


def recall_at_k(hits: List[Dict], gold: List[str], k: int) -> float:
    cand = hits[:k]
    gold = [g.lower() for g in gold]
    for h in cand:
        t = (h.get("text", "") or "").lower()
        if any(g in t for g in gold):
            return 1.0
    return 0.0


def mrr(hits: List[Dict], gold: List[str]) -> float:
    gold = [g.lower() for g in gold]
    for i, h in enumerate(hits, 1):
        t = (h.get("text", "") or "").lower()
        if any(g in t for g in gold):
            return 1.0 / i
    return 0.0


def ndcg_at_k(hits: List[Dict], gold: List[str], k: int) -> float:
    gold = [g.lower() for g in gold]

    def rel(text: str) -> int:
        t = (text or "").lower()
        return 1 if any(g in t for g in gold) else 0

    dcg = 0.0
    for i, h in enumerate(hits[:k], 1):
        dcg += rel(h.get("text", "")) / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(hits)) + 1))
    return (dcg / ideal) if ideal > 0 else 0.0
