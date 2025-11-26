import numpy as np
from rapidfuzz import fuzz


def confidence(query: str, hits: list):
    if not hits:
        return 0.0, {"reason": "no_hits"}
    texts = [h.get("text", "") for h in hits if h.get("text")]
    lex = fuzz.token_set_ratio(query, texts[0] if texts else "") / 100.0
    scores = [float(h.get("score", h.get("score_norm", 0.0))) for h in hits[:5]]
    margin = float(scores[0] - np.mean(scores[1:])) if len(scores) > 1 else scores[0]
    return 0.5 * lex + 0.5 * max(0.0, margin), {"lex": lex, "margin": margin}


def apply_crag(query, retrieve_fn, fallbacks, thresholds):
    hits = retrieve_fn(query)
    conf, dbg = confidence(query, hits)
    if conf >= max(
        thresholds["min_lexical_overlap"], thresholds["min_semantic_margin"]
    ):
        return hits, {"conf": conf, "path": "primary"}
    for fb in fallbacks:
        if fb == "increase_k":
            hits = retrieve_fn(query, top_k=32)
        elif fb == "bm25_only":
            hits = retrieve_fn(query, mode="bm25")
        elif fb == "graph_walk":
            hits = retrieve_fn(query, mode="graph")
        elif fb == "expand_query":
            hits = retrieve_fn(query, mode="expanded")
        conf, _ = confidence(query, hits)
        if conf >= min(thresholds.values()):
            return hits, {"conf": conf, "path": fb}
    return hits, {"conf": conf, "path": "low_conf"}
